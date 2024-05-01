from modules.text_generation import *
from modules.callbacks import (
    Iteratorize,
    Stream,
    _StopEverythingStoppingCriteria
)
print('Current model type:', shared.model.__class__.__name__)

shared.tokenizer.pad_token_id = shared.tokenizer.eos_token_id
shared.tokenizer.pad_token = shared.tokenizer.eos_token
shared.tokenizer.padding_side = "left" 

def encode(prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
    return shared.tokenizer(prompt, return_tensors='pt', add_special_tokens=add_special_tokens, padding=True)


def generate_reply_HF(question, original_question, seed, state, stopping_strings=None, is_chat=False):
    generate_params = {}
    for k in ['max_new_tokens', 'temperature', 'temperature_last', 'dynamic_temperature', 'dynatemp_low', 'dynatemp_high', 'dynatemp_exponent', 'smoothing_factor', 'smoothing_curve', 'top_p', 'min_p', 'top_k', 'repetition_penalty', 'presence_penalty', 'frequency_penalty', 'repetition_penalty_range', 'typical_p', 'tfs', 'top_a', 'guidance_scale', 'penalty_alpha', 'mirostat_mode', 'mirostat_tau', 'mirostat_eta', 'do_sample', 'encoder_repetition_penalty', 'no_repeat_ngram_size']:
        if k in state:
            generate_params[k] = state[k] 
            
    if isinstance(state['sampler_priority'], list) and len(state['sampler_priority']) > 0:
        generate_params['sampler_priority'] = state['sampler_priority']
    elif isinstance(state['sampler_priority'], str) and state['sampler_priority'].strip() != '':
        generate_params['sampler_priority'] = [x.strip() for x in state['sampler_priority'].replace('\n', ',').split(',') if x.strip()]

    if state['negative_prompt'] != '':
        generate_params['negative_prompt_ids'] = encode(state['negative_prompt'])

    if state['prompt_lookup_num_tokens'] > 0:
        generate_params['prompt_lookup_num_tokens'] = state['prompt_lookup_num_tokens']

    for k in ['epsilon_cutoff', 'eta_cutoff']:
        if state[k] > 0:
            generate_params[k] = state[k] * 1e-4

    if state['ban_eos_token']:
        generate_params['suppress_tokens'] = [shared.tokenizer.eos_token_id]

    if state['custom_token_bans']:
        to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
        if len(to_ban) > 0:
            if generate_params.get('suppress_tokens', None):
                generate_params['suppress_tokens'] += to_ban
            else:
                generate_params['suppress_tokens'] = to_ban

    generate_params.update({'use_cache': not shared.args.no_cache})
    if shared.args.deepspeed:
        generate_params.update({'synced_gpus': True})

    # Encode the input
    encoding = encode([question,'New prompt test'], add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
    input_ids = encoding['input_ids'].cuda()
    
    output = input_ids
    cuda = not any((shared.args.cpu, shared.args.deepspeed))
    if state['auto_max_new_tokens']:
        generate_params['max_new_tokens'] = state['truncation_length'] - input_ids.shape[-1]

    # Add the encoded tokens to generate_params
    question, input_ids, inputs_embeds = apply_extensions('tokenizer', state, question, input_ids, None)
    original_input_ids = input_ids
    
    # generate_params.update({'input_ids': input_ids})
    generate_params.update({'inputs': input_ids})
    if inputs_embeds is not None:
        generate_params.update({'inputs_embeds': inputs_embeds})

    # Stopping criteria / eos token
    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    generate_params['eos_token_id'] = eos_token_ids
    generate_params['stopping_criteria'] = transformers.StoppingCriteriaList()
    
    # Bypass stopping criteria to stop streaming for easier tests.
    # generate_params['stopping_criteria'].append(_StopEverythingStoppingCriteria())
    state['stream'] = False

    # Logits processor
    processor = state.get('logits_processor', LogitsProcessorList([]))
    if not isinstance(processor, LogitsProcessorList):
        processor = LogitsProcessorList([processor])

    # Grammar
    if state['grammar_string'].strip() != '':
        grammar = initialize_grammar(state['grammar_string'])
        grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
        processor.append(grammar_processor)

    
    apply_extensions('logits_processor', processor, input_ids)
    generate_params['logits_processor'] = processor

    if shared.args.verbose:
        logger.info("GENERATE_PARAMS=")
        filtered_params = {key: value for key, value in generate_params.items() if not isinstance(value, torch.Tensor)}
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(filtered_params)
        print()

        logger.info("PROMPT=")
        print(decode(input_ids[0], skip_special_tokens=False))
        print()

    # Handle StreamingLLM for llamacpp_HF
    if shared.model.__class__.__name__ == 'LlamacppHF' and shared.args.streaming_llm:
        # TODO this will have to be different, stored in a dict?
        tmp = process_llamacpp_cache(shared.model.model, input_ids[-1].tolist(), shared.model.model._input_ids.tolist())
        shared.model.past_seq = torch.tensor(tmp)
        shared.model.save_cache()

    t0 = time.time()
    try:
        if not is_chat and not shared.is_seq2seq:
            yield ''
            

        # Generate the entire reply at once.
        if not state['stream']:
            with torch.no_grad():
                output = shared.model.generate(**generate_params)
                print(output)
                # output = output[0]
                if cuda:
                    output = output.cuda()
                    

            starting_from = 0 if shared.is_seq2seq else len(input_ids[0])
            yield get_reply_from_output_ids(output, state, starting_from=starting_from)

        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator.
        else:
            def generate_with_callback(callback=None, *args, **kwargs):
                kwargs['stopping_criteria'].append(Stream(callback_func=callback))
                clear_torch_cache()
                with torch.no_grad():
                    shared.model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, [], kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                cumulative_reply = ''
                starting_from = 0 if shared.is_seq2seq else len(input_ids[0])
                for output in generator:
                    if output[-1] in eos_token_ids:
                        break

                    new_content = get_reply_from_output_ids(output, state, starting_from=starting_from)
                    # check the partial unicode character
                    if chr(0xfffd) in new_content:
                        continue

                    cumulative_reply += new_content
                    starting_from = len(output)
                    yield cumulative_reply

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        original_tokens = len(original_input_ids[0])
        new_tokens = len(output) - (original_tokens if not shared.is_seq2seq else 0)
        
        print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return
    

def get_reply_from_output_ids(output_ids, state=None, starting_from=0):
    reply = shared.tokenizer.batch_decode(output_ids[:,starting_from:], skip_special_tokens=state['skip_special_tokens'] if state else True)
    # reply = decode(output_ids[:,starting_from:], state['skip_special_tokens'] if state else True)
    for i in reply:
        print(f'{i!r}')

    reply = reply[0]
    # Handle tokenizers that do not add the leading space for the first token
    if (hasattr(shared.tokenizer, 'convert_ids_to_tokens') and len(output_ids) > starting_from) and not reply.startswith(' '):
        first_token = shared.tokenizer.convert_ids_to_tokens(int(output_ids[starting_from]))
        if isinstance(first_token, (bytes,)):
            first_token = first_token.decode('utf8')

        if first_token.startswith('▁'):
            reply = ' ' + reply

    return reply
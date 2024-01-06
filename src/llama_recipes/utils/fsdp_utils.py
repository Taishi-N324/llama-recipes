
def fsdp_auto_wrap_policy(model, transformer_layer_name):
    import functools
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
    from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder  # type: ignore

    def lambda_policy_fn(module):
        print(f"Checking lambda policy for module: {module}")
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            print("Lambda policy: True")
            return True
        print("Lambda policy: False")
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(
            PrefixEncoder,
            PromptEncoder,
            PromptEmbedding,
            transformer_layer_name,
            # FullyShardedDataParallelPlugin.get_module_class_from_name(
            #     model, transformer_layer_name
            # ),
        ),
    )

    print(f"Lambda policy set: {lambda_policy}")
    # print(f"Transformer wrap policy set for: {transformer_layer_cls}")

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    print(f"Auto wrap policy created with policies: {auto_wrap_policy}")

    return auto_wrap_policy
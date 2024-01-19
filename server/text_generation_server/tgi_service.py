from pathlib import Path
from loguru import logger
from text_generation_server import server
import argparse
import inspect
import optimum.habana.transformers.modeling_utils
import modeling_llama
import transformers.models.llama.modeling_llama

def main(args):
    logger.info("TGIService: starting tgi service .... ")
    logger.info(
        "TGIService: --model_id {}, --revision {}, --sharded {}, --dtype {}, --uds_path {} ".format(
            args.model_id, args.revision, args.sharded, args.dtype, args.uds_path
        )
    )
    original_adapt = optimum.habana.transformers.modeling_utils.adapt_transformers_to_gaudi

    def wrapper(*args, **kwargs):
        members = inspect.getmembers(modeling_llama)

        out = original_adapt(*args, **kwargs)

        for m in filter(lambda m: m[0].startswith("GaudiLlama") and inspect.isclass(m[1]), members):
            class_name = m[0][len('Gaudi'):]
            if cls := getattr(transformers.models.llama.modeling_llama, class_name, None):
                assert inspect.isclass(cls)
                setattr(transformers.models.llama.modeling_llama, class_name, m[1])

        return out


    optimum.habana.transformers.modeling_utils.adapt_transformers_to_gaudi = wrapper

    server.serve(
        model_id=args.model_id, revision=args.revision, dtype=args.dtype, uds_path=args.uds_path, sharded=args.sharded
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--revision", type=str)
    parser.add_argument("--sharded", type=bool)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--uds_path", type=Path)
    args = parser.parse_args()
    main(args)

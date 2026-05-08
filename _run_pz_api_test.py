"""Throwaway script to run pettingzoo's parallel_api_test against TowerFallNoArrows."""
import sys
import traceback

from pettingzoo.test import parallel_api_test

from petting_zoo_env import TowerFallNoArrows


def main():
    print("== Constructing env ==", flush=True)
    try:
        env = TowerFallNoArrows()
    except Exception:
        print("Construction failed:", flush=True)
        traceback.print_exc()
        sys.exit(1)

    print("== Running parallel_api_test (num_cycles=10) ==", flush=True)
    try:
        parallel_api_test(env, num_cycles=10)
    except Exception:
        print("parallel_api_test failed:", flush=True)
        traceback.print_exc()
        sys.exit(2)
    finally:
        try:
            env.close()
        except Exception:
            pass

    print("== PASSED ==", flush=True)


if __name__ == "__main__":
    main()

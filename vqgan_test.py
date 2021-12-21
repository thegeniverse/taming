from taming.modeling_utils import load_model


def test():
    try:
        model = load_model()
        print("OK 👌")

    except Exception as e:
        print("ERROR! 😥")
        print(e)


if __name__ == "__main__":
    test()

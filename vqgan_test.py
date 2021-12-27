from taming.modeling_utils import load_model


def test():
    try:
        model_name = "openimages-8192"
        model = load_model(model_name=model_name, )
        print("OK 👌")

    except Exception as e:
        print("ERROR! 😥")
        print(e)


if __name__ == "__main__":
    test()

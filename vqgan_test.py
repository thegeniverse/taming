from modeling_utils import load_model

try:
    model = load_model()
    print("OK 👌")

except Exception as e:
    print("ERROR! 😥")
    print(e)

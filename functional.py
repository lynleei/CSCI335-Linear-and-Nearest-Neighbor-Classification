def func_generator():
    print("outer function")
    def inner_function():
        print("inner function")
    return inner_function

func = func_generator()
func()

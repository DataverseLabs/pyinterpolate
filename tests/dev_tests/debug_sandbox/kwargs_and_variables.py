def check_kwargs(x, y0, **kwargs):

    dd = {
        "x": x,
        "y": y0
    }

    if kwargs:
        dd.update(kwargs)

    print(dd)


if __name__ == '__main__':
    check_kwargs(1, 2)
    k1 = {"k": 3}
    check_kwargs(1, 2, **k1)
    k2 = {"k": 4, "w": 8}
    check_kwargs(1, 2, **k2)
def CNN_output_shape(
    input_size: int = 188,
    dilation: int = 1,
    kernel_size: int = 3,
    padding: int = 0,
    stride: int = 1,
) -> int:
    output = int(
        ((input_size + 2 * padding - (dilation * (kernel_size - 1)) - 1) / stride) + 1
    )

    return output


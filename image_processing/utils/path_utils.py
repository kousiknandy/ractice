from pathlib import PurePath, Path

OUTPUT = "output"


def walk_files(top_dir, filters):
    path = Path(top_dir)
    for f in filters:
        yield from path.rglob(f, case_sensitive=False)


def get_output_path(image_path: str, transform_path: str):
    ip = PurePath(image_path)
    tp = PurePath(transform_path)
    op = OUTPUT / PurePath(*list(ip.parent.parts)[1:])
    fn = "-".join([ip.stem] + list(tp.parts)[1:-1] + [tp.stem]) + ip.suffix
    return str(op / fn)


if __name__ == "__main__":
    assert (
        get_output_path("images/x11/zx.png", "transforms/flip.json")
        == OUTPUT + "/x11/zx-flip.png"
    ), get_output_path("images/x11/zx.png", "transforms/flip.json")

    assert (
        get_output_path("images/x11/x12/x13/zx.png", "transforms/flip.json")
        == OUTPUT + "/x11/x12/x13/zx-flip.png"
    ), get_output_path("images/x11/x12/x13/zx.png", "transforms/flip.json")

    assert (
        get_output_path("images/x11/zx.png", "transforms/x12/flip.json")
        == OUTPUT + "/x11/zx-x12-flip.png"
    ), get_output_path("images/x11/zx.png", "transforms/x12/flip.json")

    assert (
        get_output_path("images/x11/x12/zx.png", "transforms/x13/flip.json")
        == OUTPUT + "/x11/x12/zx-x13-flip.png"
    ), get_output_path("images/x11/x12/zx.png", "transforms/x13/flip.json")

    assert (
        get_output_path("images/zx.png", "transforms/flip.json")
        == OUTPUT + "/zx-flip.png"
    ), get_output_path("images/x11/x12/zx.png", "transforms/x13/flip.json")

    assert (
        get_output_path("images/x11/zx.jpg", "transforms/flip.json")
        == OUTPUT + "/x11/zx-flip.jpg"
    ), get_output_path("images/x11/zx.jpg", "transforms/flip.json")

    assert (
        get_output_path("images/x11/zx.png", "transforms/flip.yaml")
        == OUTPUT + "/x11/zx-flip.png"
    ), get_output_path("images/x11/zx.png", "transforms/flip.yaml")

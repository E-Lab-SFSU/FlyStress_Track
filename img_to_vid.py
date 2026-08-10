from pathlib import Path
import argparse
import subprocess
import tempfile


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_images(folder):
    folder = Path(folder)

    return sorted(
        path for path in folder.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def create_video(images, output, duration, width, height):
    if not images:
        raise ValueError("No images found.")

    with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
    ) as file_list:

        for image in images:
            path = image.resolve().as_posix()
            file_list.write(f"file '{path}'\n")
            file_list.write(f"duration {duration}\n")

        # FFmpeg concat requires the final image again.
        file_list.write(f"file '{images[-1].resolve().as_posix()}'\n")

        list_path = file_list.name

    video_filter = (
        f"scale={width}:{height}:"
        "force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        "format=yuv420p"
    )

    command = [
        "C:/Users/chana/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0-full_build/bin/ffmpeg.exe",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-vf", video_filter,
        "-r", "30",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output,
    ]

    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a folder of images into an MP4 video."
    )

    parser.add_argument("folder")
    parser.add_argument("output")
    parser.add_argument("--duration", type=float, default=1)        # change video speed. < 1 is faster, > 1 is slower
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)

    args = parser.parse_args()

    images = get_images(args.folder)

    print(f"Found {len(images)} images.")

    create_video(
        images,
        args.output,
        args.duration,
        args.width,
        args.height,
    )

    print(f"Created: {args.output}")


if __name__ == "__main__":
    main()
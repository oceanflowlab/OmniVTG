"""
OmniVTG Dataset Video Download Script
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def check_yt_dlp():
    """Verify yt-dlp is installed and accessible."""
    try:
        subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            check=True,
            text=True,
        )
    except FileNotFoundError:
        print("Error: 'yt-dlp' is not installed. Install it with: pip install yt-dlp")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("Error: 'yt-dlp' is installed but returned an error. Check your installation.")
        sys.exit(1)


def build_url_and_platform(vid):
    """Build download URL and determine platform from vid."""
    if vid.startswith("v_"):
        youtube_id = vid[2:]
        return f"https://www.youtube.com/watch?v={youtube_id}", "youtube"
    elif vid.startswith("BV"):
        return f"https://www.bilibili.com/video/{vid}", "bilibili"
    else:
        raise ValueError(f"Unknown vid format (expected 'v_' or 'BV' prefix): {vid}")


def download_one_video(vid, video_dir, yt_cookies, bili_cookies, tries, timeout):
    """Download a single video. Returns (vid, status, message)."""
    url, platform = build_url_and_platform(vid)
    output_path = video_dir / f"{vid}.mp4"

    if output_path.exists():
        return vid, "skipped", "already exists"

    # Select cookies based on platform
    cookies = yt_cookies if platform == "youtube" else bili_cookies

    # Build yt-dlp command
    cmd = ["yt-dlp"]

    if platform == "youtube":
        cmd.extend([
            "-f",
            "bestvideo[ext=mp4][height<=720][fps<=30]"
            "/best[ext=mp4][height<=720][fps<=30]"
            "/best[height<=720][fps<=30]"
            "/best",
        ])
    else:
        cmd.extend([
            "-f",
            "bestvideo[height<=720][fps<=30]"
            "/best[height<=720][fps<=30]"
            "/best",
        ])

    cmd.extend([
        "--output", str(output_path),
        "--retries", str(tries),
        "--fragment-retries", str(tries),
        "--no-write-info-json",
        "--no-write-thumbnail",
        "--no-write-description",
        "--no-write-comments",
        "--no-download-archive",
        "--ignore-errors",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--remote-components", "ejs:github",
    ])

    if cookies:
        cmd.extend(["--cookies", cookies])

    cmd.append(url)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode == 0 and output_path.exists():
            return vid, "downloaded", "OK"
        else:
            stderr = result.stderr.strip() if result.stderr else ""
            err_msg = f"returncode={result.returncode}, stderr={stderr[:300]}"
            return vid, "failed", err_msg

    except subprocess.TimeoutExpired:
        return vid, "failed", "timeout"
    except Exception as e:
        return vid, "failed", str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Download OmniVTG dataset videos."
    )
    parser.add_argument(
        "raw_data",
        type=Path,
        help="Path to raw_data.json containing video annotations.",
    )
    parser.add_argument(
        "--youtube-cookies",
        type=Path,
        default=None,
        help="Path to YouTube cookies file in Netscape format.",
    )
    parser.add_argument(
        "--bilibili-cookies",
        type=Path,
        default=None,
        help="Path to BiliBili cookies file in Netscape format.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("videos"),
        help="Directory to save downloaded videos (default: videos/).",
    )
    parser.add_argument(
        "--tries",
        type=int,
        default=5,
        help="Number of retries per video (default: 5).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent downloads (default: 4).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Download timeout in seconds per video (default: 600).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of videos to process (useful for testing).",
    )

    args = parser.parse_args()

    # Pre-flight: check yt-dlp
    check_yt_dlp()

    # Load video list
    if not args.raw_data.exists():
        print(f"Error: {args.raw_data} not found.")
        sys.exit(1)

    with open(args.raw_data, "r", encoding="utf-8") as f:
        data = json.load(f)

    vids = list(data.keys())
    if args.limit:
        vids = vids[:args.limit]

    yt_count = sum(1 for v in vids if v.startswith("v_"))
    bili_count = sum(1 for v in vids if v.startswith("BV"))

    print(f"Total videos: {len(vids)} (YouTube: {yt_count}, BiliBili: {bili_count})")

    # Prepare
    args.video_dir.mkdir(parents=True, exist_ok=True)

    yt_cookies = str(args.youtube_cookies) if args.youtube_cookies else None
    bili_cookies = str(args.bilibili_cookies) if args.bilibili_cookies else None

    if yt_cookies:
        print(f"Using YouTube cookies: {args.youtube_cookies}")
    if bili_cookies:
        print(f"Using BiliBili cookies: {args.bilibili_cookies}")

    # Download
    results = {"downloaded": 0, "skipped": 0, "failed": 0}
    failed_vids = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for vid in vids:
            fut = executor.submit(
                download_one_video,
                vid,
                args.video_dir,
                yt_cookies,
                bili_cookies,
                args.tries,
                args.timeout,
            )
            futures[fut] = vid

        for fut in as_completed(futures):
            vid = futures[fut]
            try:
                vid, status, msg = fut.result()
                results[status] += 1
                if status == "failed":
                    failed_vids.append((vid, msg))
                    if args.limit:
                        print(msg)
            except Exception as e:
                results["failed"] += 1
                failed_vids.append((vid, str(e)))

            # Progress line
            total = len(vids)
            done = results["downloaded"] + results["skipped"] + results["failed"]
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            print(
                f"[{done}/{total}] "
                f"down: {results['downloaded']} | "
                f"skip: {results['skipped']} | "
                f"fail: {results['failed']} | "
                f"{rate:.1f} vids/s"
            )

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Finished in {elapsed:.0f}s")
    print(f"  Downloaded: {results['downloaded']}")
    print(f"  Skipped:    {results['skipped']}")
    print(f"  Failed:     {results['failed']}")

    if failed_vids:
        print(f"\nFailed videos ({len(failed_vids)}):")
        for vid, reason in failed_vids[:20]:
            print(f"  {vid}: {reason}")
        if len(failed_vids) > 20:
            print(f"  ... and {len(failed_vids) - 20} more")

        # Write failures to file
        fail_log = Path("download_failures.txt")
        with open(fail_log, "w", encoding="utf-8") as f:
            for vid, reason in failed_vids:
                f.write(f"{vid}\t{reason}\n")
        print(f"\nFull failure list written to {fail_log}")


if __name__ == "__main__":
    main()

# OmniVTG Dataset

## Annotations and Data Preparation

We provide the annotation on [OmniVTG-Dataset](https://huggingface.co/datasets/zhengmh/OmniVTG-Dataset).

Please move all the files in `OmniVTG-Dataset/train_data` into `OmniVTG/`.

Please move all the files in `OmniVTG-Dataset/test_data` into `../standalone_eval/annotations/`.

## Video Download

Since raw videos cannot be directly distributed, we provide a download script to fetch all videos from their original platforms.

### Dependencies

- **Python 3.8+**
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** — video downloader, install with:

  ```bash
  pip install yt-dlp
  ```

### Usage

```bash
python download_videos.py <path_to_raw_data.json> [options]
```

#### Required Arguments

| Argument       | Description                          |
| -------------- | ------------------------------------ |
| `raw_data`     | Path to the `raw_data.json` file in `OmniVTG-Dataset`.    |

#### Optional Arguments

| Option                  | Default    | Description                                              |
| ----------------------- | ---------- | -------------------------------------------------------- |
| `--youtube-cookies`     | None       | Path to Netscape-format cookies file for YouTube.         |
| `--bilibili-cookies`    | None       | Path to Netscape-format cookies file for BiliBili.        |
| `--video-dir`           | `videos`   | Directory to save downloaded videos.                      |
| `--tries`               | `5`        | Number of retries per video on failure.                   |
| `--workers`             | `4`        | Number of concurrent download threads.                    |
| `--timeout`             | `600`      | Timeout in seconds for each video download.               |
| `--limit`               | None       | Limit the number of videos to process (useful for testing). |

#### Examples

```bash
# Download all videos
python download_videos.py raw_data.json

# Download with cookies (recommended for better access)
python download_videos.py raw_data.json \
    --youtube-cookies youtube_cookies.txt \
    --bilibili-cookies bilibili_cookies.txt

# Test with a small subset
python download_videos.py raw_data.json --limit 10

# Increase concurrency for faster downloads
python download_videos.py raw_data.json --workers 8
```

### Cookie Files

Cookies help yt-dlp access videos that may be region-restricted or require login. Both cookie files should follow the **Netscape HTTP Cookie File format**.

**How to obtain cookies:**

1. Install a browser extension such as [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) for Chrome/Edge, or [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/) for Firefox.
2. Log in to [YouTube](https://www.youtube.com) and/or [BiliBili](https://www.bilibili.com).
3. Use the extension to export cookies in Netscape format.
4. Pass the exported `.txt` files via `--youtube-cookies` and `--bilibili-cookies`.

### Output

All videos are saved to `videos/` with the naming convention `{vid}.mp4`, where `vid` is the key from `raw_data.json`:

- `videos/v_B6QMxTYWS3E.mp4` — YouTube video (`v_` prefix)
- `videos/BV12e4y1z7x3.mp4` — BiliBili video (`BV` prefix)

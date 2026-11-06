def save_annotated_video(results, output_path, fps):
    writer = None
    frames_written = 0
    try:
        for result in results:
            frame = result.plot()
            if writer is None:
                height, width = frame.shape[:2]
                writer = imageio_ffmpeg.write_frames(
                    str(output_path), (width, height), fps=float(fps),
                    codec="libx264", pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
                    quality=7, macro_block_size=2, ffmpeg_log_level="error",
                    output_params=["-movflags", "+faststart"]
                )
                writer.send(None)
            writer.send(np.ascontiguousarray(frame))
            frames_written += 1
    finally:
        if writer is not None:
            writer.close()
        if hasattr(results, "close"):
            results.close()
    assert frames_written > 0, "No video frames were processed"
    assert output_path.is_file() and output_path.stat().st_size > 0, "Video was not saved"
    decoded_frames, duration = imageio_ffmpeg.count_frames_and_secs(str(output_path))
    assert decoded_frames == frames_written, (decoded_frames, frames_written)
    assert duration > 0, "Invalid video duration"
    return frames_written, duration

OUTPUT_VIDEO = WORK / "detection.mp4"
video_results = best_model.predict(
    source=str(video_path), stream=True, vid_stride=1,
    imgsz=640, conf=0.25, iou=0.7, device=0,
    save=False, verbose=False
)
frames_written, duration = save_annotated_video(video_results, OUTPUT_VIDEO, video_fps)
if video_frame_count > 0:
    assert frames_written == video_frame_count, (frames_written, video_frame_count)
print({"output": str(OUTPUT_VIDEO), "frames": frames_written,
       "seconds": duration, "bytes": OUTPUT_VIDEO.stat().st_size})
display(Video(str(OUTPUT_VIDEO), embed=True, width=800))
display(FileLink("detection.mp4"))

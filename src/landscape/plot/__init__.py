from logging import warning

def _animate_progress(current_frame, total_frames):
    print("\r" + f"Processing {current_frame+1}/{total_frames} frames...", end="")
    if current_frame + 1 == total_frames:
        print("\nConverting to gif, this may take a while...")

def sample_frames(steps, max_frames):
    samples = []
    steps_len = len(steps)
    if max_frames > steps_len:
        warning(f"Less than {max_frames} frames provided, producing {steps_len} frames.")
        max_frames = steps_len
    interval = steps_len // max_frames
    counter = 0
    for i in range(steps_len - 1, -1, -1):  # Sample from the end
        if i % interval == 0 and counter < max_frames:
            samples.append(steps[i])
            counter += 1

    return list(reversed(samples))
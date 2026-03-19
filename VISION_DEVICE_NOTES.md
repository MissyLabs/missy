# Vision Device Notes

## Target Camera: Logitech C922x Pro Stream

### Specifications

| Property | Value |
|----------|-------|
| Model | Logitech C922x Pro Stream Webcam |
| USB Vendor ID | `046d` |
| USB Product ID | `085c` (C922x), `085b` (C922) |
| Max Resolution | 1920x1080 @ 30fps, 1280x720 @ 60fps |
| Connection | USB 2.0 (Type-A) |
| Driver | UVC (USB Video Class) — native Linux support |
| V4L2 | Supported, no additional drivers needed |

### Linux Setup

```bash
# Verify USB detection
lsusb | grep -i logitech
# Expected: Bus 001 Device 002: ID 046d:085c Logitech, Inc. C922x Pro Stream Webcam

# Verify V4L2 device
ls -la /dev/video*
# Expected: /dev/video0 (capture), /dev/video1 (metadata)

# Verify user permissions
groups
# Must include 'video' group

# Add user to video group if needed
sudo usermod -aG video $USER
# Log out and back in for group change to take effect

# Check sysfs
cat /sys/class/video4linux/video0/name
# Expected: Logitech C922x Pro Stream Webcam
```

### Known Issues

1. **Dual device nodes**: The C922x creates two `/dev/video*` entries — `video0` (capture) and `video1` (metadata). Missy's discovery filters by sysfs `index=0` to select the capture node only.

2. **Auto-exposure warm-up**: Initial frames may be dark or color-shifted. Missy discards 5 warm-up frames by default.

3. **Resolution fallback**: If 1920x1080 is not available (e.g., bandwidth-limited USB hub), the camera may negotiate a lower resolution. Missy accepts whatever the camera provides.

4. **USB hub compatibility**: Some USB 3.0 hubs may cause bandwidth issues. Connect directly to a motherboard USB port for best results.

## Other Supported Cameras

The discovery system works with any UVC-compatible USB camera. Known models in the database:

| USB ID | Model |
|--------|-------|
| `046d:085c` | Logitech C922x Pro Stream |
| `046d:085b` | Logitech C922 Pro Stream |
| `046d:0825` | Logitech HD Webcam C270 |
| `046d:082d` | Logitech HD Pro Webcam C920 |
| `046d:0843` | Logitech Webcam C930e |

Any USB camera not in this list will still be discovered and usable — it just won't be auto-preferred.

## Device Path Stability

Linux assigns `/dev/videoN` paths based on enumeration order, which can change:
- After reboot
- When USB devices are reconnected
- When other video devices are added/removed

Missy handles this by:
1. Identifying cameras by USB vendor/product ID (stable across reboots)
2. Reading the sysfs `device` symlink to find the USB topology
3. Caching discovery results with TTL-based invalidation
4. Re-discovering on cache miss or forced refresh

## Permissions

Camera access requires:
- User membership in the `video` group
- Read/write access to `/dev/videoN` device nodes
- Read access to `/sys/class/video4linux/` (world-readable by default)

The `missy vision doctor` command checks all of these prerequisites.

## Performance Notes

- First capture takes ~200ms (device open + warm-up)
- Subsequent captures within same session: ~50ms
- Image preprocessing (resize + CLAHE): ~20ms for 1920x1080
- JPEG encoding for LLM: ~10ms
- Scene memory change detection: ~5ms per comparison

## Troubleshooting

### "No cameras detected"
1. Check USB connection: `lsusb`
2. Check device nodes: `ls /dev/video*`
3. Check permissions: `ls -la /dev/video0`
4. Run `missy vision doctor` for full diagnostics

### "Cannot open camera"
1. Another application may be using the camera
2. Check with `fuser /dev/video0`
3. Close other applications using the camera

### "Blank frames"
1. Camera may need warm-up time — increase `warmup_frames`
2. Lens cap may be on
3. Low light conditions — check with `missy vision inspect`

### "Permission denied"
```bash
sudo usermod -aG video $USER
# Then log out and back in
```

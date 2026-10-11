# Landing-screen background

Save the RADFRAC home-screen backdrop here as:

```
app/assets/home_bg.jpg
```

`.png`, `.jpeg`, `.webp` and `.bmp` also work, and any filename containing
`bg`, `hero` or `background` is picked up as well (see
`theme.find_background_file`).

The image is inlined as a data URI by `app/theme.py`, downscaled to 1800 px on
the long edge if it is larger than ~350 KB, and washed to near-white before it is
drawn so body text keeps AA contrast. The cache key includes the file's
modification time and size, so replacing the file takes effect on the next rerun.

If no image is present the theme falls back to a CSS-only light gradient and the
app still renders correctly.

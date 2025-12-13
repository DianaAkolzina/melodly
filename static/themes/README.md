Theme files

- Put CSS files in this folder to override UI variables.
- Each theme should define CSS variables on :root, e.g.:

  :root {
    --lemongrass: #9dc183;
    --lavender: #b7a9e0;
    --ink: #2c2c2c;
    --bg: #f7f8f7;
    --card: #ffffff;
    --accent: var(--lemongrass);
    --accent-2: var(--lavender);
  }

How to load a theme

- In the app, call setTheme('<file-basename>') from the JS console or wire it to a control.
- Example: setTheme('sample-lemongrass') loads /static/themes/sample-lemongrass.css

Shapes & decorative assets

- Place vector shapes in static/assets/shapes/ (SVG preferred):
  - static/assets/shapes/blob-1.svg
  - static/assets/shapes/wave-1.svg
- Reference them in CSS using background-image on a positioned element, e.g.:

  .shape.blob {
    position: absolute; inset: auto -80px -80px auto; width: 240px; height: 240px;
    background: url('/static/assets/shapes/blob-1.svg') no-repeat center / contain;
    opacity: .15; pointer-events: none;
  }

- Add the element in templates/index.html near the app container if desired:

  <div class="shape blob" aria-hidden="true"></div>


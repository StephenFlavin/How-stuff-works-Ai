# How-stuff-works-Ai

**Apple in space** — an interactive explainer for how large language models turn
words into vectors, mix them with attention, stack them through transformer
blocks, and generate text one token at a time.

🔗 **Live site:** https://stephenflavin.github.io/How-stuff-works-Ai/

## Tech

A single-page [Vite](https://vitejs.dev/) + [React](https://react.dev/) app. The
whole explainer lives in [`src/App.jsx`](src/App.jsx).

## Develop locally

```bash
npm install
npm run dev      # start the dev server
npm run build    # production build into dist/
npm run preview  # preview the production build
```

## Deployment

Pushing to `main` triggers the
[`Deploy to GitHub Pages`](.github/workflows/deploy.yml) workflow, which builds
the site and publishes `dist/` to GitHub Pages.

**One-time setup:** in the repository **Settings → Pages**, set **Source** to
**GitHub Actions**. After that, every push to `main` redeploys automatically.

> The Vite `base` is set to `"./"` (relative) in
> [`vite.config.js`](vite.config.js), so assets resolve correctly under the
> case-sensitive project-pages path regardless of the repo name's casing.

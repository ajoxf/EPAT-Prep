# Deployment Guide

This guide covers deploying your Quantitative Finance LMS to various platforms.

## Prerequisites

Before deploying, ensure:
- All tests pass locally (`npm run dev` works)
- Build completes successfully (`npm run build`)
- All dependencies are in package.json

## Option 1: GitHub Pages (Free)

### Setup

1. **Install gh-pages package:**
```bash
npm install --save-dev gh-pages
```

2. **Update package.json:**
```json
{
  "scripts": {
    "predeploy": "npm run build",
    "deploy": "gh-pages -d dist"
  },
  "homepage": "https://yourusername.github.io/quant-finance-lms"
}
```

3. **Update vite.config.js:**
```javascript
export default defineConfig({
  plugins: [react()],
  base: '/quant-finance-lms/', // Your repository name
})
```

4. **Deploy:**
```bash
npm run deploy
```

5. **Enable GitHub Pages:**
   - Go to repository Settings
   - Navigate to Pages section
   - Select `gh-pages` branch
   - Save

Your app will be live at `https://yourusername.github.io/quant-finance-lms`

## Option 2: Netlify (Free)

### Method A: Netlify CLI

1. **Install Netlify CLI:**
```bash
npm install -g netlify-cli
```

2. **Build your project:**
```bash
npm run build
```

3. **Deploy:**
```bash
netlify deploy --prod --dir=dist
```

### Method B: Git Integration

1. Push your code to GitHub
2. Go to [Netlify](https://netlify.com)
3. Click "Add new site" → "Import an existing project"
4. Choose GitHub and select your repository
5. Configure build settings:
   - **Build command:** `npm run build`
   - **Publish directory:** `dist`
6. Click "Deploy site"

Your app will be live at `https://your-site-name.netlify.app`

### Custom Domain on Netlify

1. Go to Site settings → Domain management
2. Click "Add custom domain"
3. Follow DNS configuration instructions

## Option 3: Vercel (Free)

### Method A: Vercel CLI

1. **Install Vercel CLI:**
```bash
npm install -g vercel
```

2. **Deploy:**
```bash
vercel
```

3. **For production:**
```bash
vercel --prod
```

### Method B: Git Integration

1. Push your code to GitHub
2. Go to [Vercel](https://vercel.com)
3. Click "Import Project"
4. Select your GitHub repository
5. Vercel auto-detects Vite settings
6. Click "Deploy"

Your app will be live at `https://your-project.vercel.app`

## Option 4: Firebase Hosting (Free Tier)

1. **Install Firebase tools:**
```bash
npm install -g firebase-tools
```

2. **Login to Firebase:**
```bash
firebase login
```

3. **Initialize Firebase:**
```bash
firebase init hosting
```

Select:
- Public directory: `dist`
- Single-page app: Yes
- Automatic builds: No

4. **Build and deploy:**
```bash
npm run build
firebase deploy
```

## Option 5: Self-Hosting

### Using a Simple HTTP Server

1. **Build the project:**
```bash
npm run build
```

2. **Install serve:**
```bash
npm install -g serve
```

3. **Run:**
```bash
serve -s dist -p 3000
```

### Using NGINX

1. Build the project
2. Copy `dist` folder to your server
3. Configure NGINX:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /path/to/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

### Using Apache

1. Build the project
2. Copy `dist` folder to web root
3. Create `.htaccess`:

```apache
<IfModule mod_rewrite.c>
  RewriteEngine On
  RewriteBase /
  RewriteRule ^index\.html$ - [L]
  RewriteCond %{REQUEST_FILENAME} !-f
  RewriteCond %{REQUEST_FILENAME} !-d
  RewriteRule . /index.html [L]
</IfModule>
```

## Environment-Specific Builds

### Development
```bash
npm run dev
```

### Production
```bash
npm run build
npm run preview  # Preview production build locally
```

## Custom Domain Setup

### For Netlify/Vercel

1. Purchase domain (e.g., from Namecheap, GoDaddy)
2. Add domain in platform settings
3. Update DNS records:
   - **A Record:** Point to platform's IP
   - **CNAME:** Point www to platform domain

### SSL Certificate

Most platforms (Netlify, Vercel, Firebase) provide free SSL certificates automatically.

## Optimization Tips

### Before Deployment

1. **Optimize images:**
   - Compress images using tools like TinyPNG
   - Use appropriate formats (WebP when possible)

2. **Code splitting:**
   - Vite does this automatically
   - Consider lazy loading for large chapters

3. **Remove console logs:**
```javascript
// Add to vite.config.js for production
export default defineConfig({
  esbuild: {
    drop: ['console', 'debugger'],
  },
})
```

4. **Analyze bundle size:**
```bash
npm run build -- --mode analyze
```

## Continuous Deployment

### GitHub Actions for Netlify

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Netlify

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
        with:
          node-version: '18'
      - run: npm ci
      - run: npm run build
      - uses: netlify/actions/cli@master
        with:
          args: deploy --prod --dir=dist
        env:
          NETLIFY_SITE_ID: ${{ secrets.NETLIFY_SITE_ID }}
          NETLIFY_AUTH_TOKEN: ${{ secrets.NETLIFY_AUTH_TOKEN }}
```

## Monitoring

### Analytics

Add Google Analytics to `index.html`:

```html
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

### Error Tracking

Consider services like:
- Sentry
- LogRocket
- Rollbar

## Troubleshooting

### Build Fails

1. Clear node_modules and reinstall:
```bash
rm -rf node_modules package-lock.json
npm install
```

2. Check Node version (should be 16+):
```bash
node --version
```

### Routes Not Working

Ensure your hosting platform is configured for single-page apps:
- All routes should serve `index.html`
- Configure rewrites/redirects appropriately

### Slow Load Times

1. Check bundle size
2. Enable compression (gzip/brotli)
3. Use CDN if possible
4. Optimize images

## Rollback

### Netlify/Vercel
- Both platforms keep deployment history
- Rollback from dashboard to previous deployment

### GitHub Pages
```bash
git revert <commit-hash>
git push origin main
npm run deploy
```

## Security Considerations

1. Keep dependencies updated:
```bash
npm audit
npm audit fix
```

2. Use HTTPS (enabled by default on most platforms)
3. Set appropriate CORS headers if needed
4. Don't commit sensitive data

## Cost Estimates

- **GitHub Pages:** Free
- **Netlify (Free tier):** 100GB bandwidth, 300 build minutes/month
- **Vercel (Free tier):** 100GB bandwidth, unlimited sites
- **Firebase (Free tier):** 10GB storage, 360MB/day transfer

Most free tiers are sufficient for educational use!

## Support

For platform-specific issues:
- GitHub Pages: GitHub Documentation
- Netlify: Netlify Support
- Vercel: Vercel Documentation
- Firebase: Firebase Support

---

Choose the platform that best fits your needs. For a simple, free solution, GitHub Pages or Netlify are excellent choices!

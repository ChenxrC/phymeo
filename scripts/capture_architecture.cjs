// Optional maintainer helper: npm-install Playwright outside the source package.
const { chromium } = require('playwright');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.PHYMEO_BROWSER_EXECUTABLE || undefined });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
    const root = path.resolve(__dirname, '..');
    await page.goto(pathToFileURL(path.join(root, 'docs/architecture/phymeo.html')).href);
    await page.screenshot({ path: path.join(root, 'docs/architecture/phymeo.png'), fullPage: true });
    const overflow = await page.evaluate(() => [...document.querySelectorAll('g[data-box]')].flatMap(g => {
      const r = g.querySelector('rect').getBBox();
      return [...g.querySelectorAll('text')].filter(t => {
        const b = t.getBBox();
        return b.x < r.x || b.x + b.width > r.x + r.width || b.y < r.y || b.y + b.height > r.y + r.height;
      }).map(t => t.textContent);
    }));
    if (overflow.length) throw new Error(`SVG labels overflow: ${overflow.join(', ')}`);
    await page.setViewportSize({ width: 390, height: 844 });
    const mobile = await page.evaluate(() => ({ page: document.documentElement.scrollWidth, viewport: innerWidth, scrollableDiagram: document.querySelector('.diagram-container').scrollWidth }));
    if (mobile.page > mobile.viewport) throw new Error(`Unexpected mobile body overflow: ${JSON.stringify(mobile)}`);
    console.log(JSON.stringify({ screenshot: 'docs/architecture/phymeo.png', desktopLabelOverflow: overflow, mobile }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

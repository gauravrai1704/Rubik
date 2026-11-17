// Records the backup video of the demo (demo/demo_recording.webm). Start the server first:
//   python demo/server.py --port 8000 &   then   node demo/record_demo.js
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
const PORT = process.env.PORT || 8000;
(async () => {
  const dir = path.join(__dirname, 'video_tmp');
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 1440, height: 860 }, recordVideo: { dir, size: { width: 1440, height: 860 } } });
  const p = await ctx.newPage();
  const wait = ms => p.waitForTimeout(ms);
  const solved = () => p.waitForFunction(() => !document.querySelector('#res').innerText.includes('running'), null, { timeout: 180000 });
  await p.goto(`http://localhost:${PORT}/`); await wait(2500);
  // 1. puzzles and clusters
  await p.check('#clu'); await wait(2500); await p.uncheck('#clu');
  await p.click('#kCu'); await wait(1500); await p.check('#clu'); await wait(2500); await p.uncheck('#clu');
  await p.click('#kSq'); await wait(1500);
  // 2. Square 6 live
  await p.click('#scr'); await wait(1500); await p.click('#solve'); await solved(); await wait(2500);
  await p.click('#res tbody tr.best button'); await p.fill('#speed', '6'); await p.click('#play'); await wait(4000);
  // Square 14
  await p.selectOption('#N', '14'); await wait(1000); await p.click('#scr'); await wait(1000); await p.click('#solve'); await solved(); await wait(3000);
  // 3. Cube 6 live
  await p.click('#kCu'); await wait(1500); await p.click('#scr'); await wait(1500); await p.click('#solve'); await solved(); await wait(2500);
  await p.click('#res tbody tr.best button'); await p.fill('#speed', '30'); await p.click('#play'); await wait(5000); await p.click('#play');
  // 4. Santa
  await p.click('#t2'); await wait(3000); await p.click('#rsym'); await wait(7500); await p.click('#ver'); await wait(3000);
  await p.selectOption('#pz', '10'); await wait(3500);
  const v = await p.video().path(); await ctx.close(); await b.close();
  fs.renameSync(v, path.join(__dirname, 'demo_recording.webm')); fs.rmSync(dir, { recursive: true, force: true });
  console.log('wrote', path.join(__dirname, 'demo_recording.webm'));
})();

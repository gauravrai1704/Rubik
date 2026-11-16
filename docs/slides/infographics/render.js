const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1300, height: 3000 }, deviceScaleFactor: 2 });
  await p.goto('file://' + __dirname + '/info.html');
  for (const id of ['pA','pB','pC','pD','pE']) await (await p.$('#' + id)).screenshot({ path: id + '.png' });
  await b.close();
})();

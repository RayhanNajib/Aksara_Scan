/**
 * Web Worker untuk AksaraEnhancer agar pemrosesan foto besar / CLAHE / Richardson-Lucy
 * berjalan sepenuhnya di luar main thread UI (tanpa blocking / long task).
 */
/* global importScripts, AksaraEnhancer */
importScripts('./image_enhancer.js');

self.onmessage = function (e) {
  const { id, op, width, height, buffer, params } = e.data || {};
  try {
    const rgba = new Uint8ClampedArray(buffer);
    if (op === 'estimateBlur') {
      const report = AksaraEnhancer.estimateBlur({ data: rgba, width, height });
      self.postMessage({ id, ok: true, report });
      return;
    }
    const res = AksaraEnhancer.enhance({ data: rgba, width, height }, params || {});
    const outBuf = res.imageData.data.buffer;
    self.postMessage({
      id,
      ok: true,
      width: res.imageData.width,
      height: res.imageData.height,
      buffer: outBuf,
      report: res.report,
    }, [outBuf]);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err) });
  }
};

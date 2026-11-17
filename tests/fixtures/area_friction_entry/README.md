# Fixed actual entry for the resident friction pack test

This is the complete unmodified I3v accepted payload from 2026-11-18, source/test commit 825902af. Original: artifacts/area-pressure-entry-friction-20261118-01/06-payloads/test_actual_entry_native_oracl0/entry-counterfactual.json.gz.

SHA256: 50583f6e9f6875a77e09c3c3fa645588aa30c33b7b7d4b6113b1d1731bccd1b6 (72010 bytes). It preserves actual device-derived dynamics, original quadratures, raw normal rows, one-step receipt, independent CPU entry oracles, and three independent friction counterfactuals. Array dtype/shape/hex retain exact stored words. All units are SI. The FREE example has an offset COM and 18 samples; the damped horizontal slide has no samples.

Reuse this fixed data to verify the new device-layout pack kernel without rerunning the accepted scene. Uploading this saved entry is a kernel seam test, not a live zero-copy production scene, not a new physical trajectory, and not a receipt authentication mechanism. Do not regenerate/tune it to make a new solver pass. See docs/development/area-pressure-entry-friction-report-20261118.md for its original scope and retained first static failures.
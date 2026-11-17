# Fixed actual entry for the resident friction pack test

This is the complete unmodified I3v accepted payload from 2026-11-18, source/test commit 825902af. Original: artifacts/area-pressure-entry-friction-20261118-01/06-payloads/test_actual_entry_native_oracl0/entry-counterfactual.json.gz.

SHA256: 50583f6e9f6875a77e09c3c3fa645588aa30c33b7b7d4b6113b1d1731bccd1b6 (72010 bytes). It preserves actual device-derived dynamics, original quadratures, raw normal rows, one-step receipt, independent CPU entry oracles, and three independent friction counterfactuals. Array dtype/shape/hex retain exact stored words. All units are SI. The FREE example has an offset COM and 18 samples; the damped horizontal slide has no samples.

Reuse this fixed data to verify the new device-layout pack kernel without rerunning the accepted scene. Uploading this saved entry is a kernel seam test, not a live zero-copy production scene, not a new physical trajectory, and not a receipt authentication mechanism. Do not regenerate/tune it to make a new solver pass. See docs/development/area-pressure-entry-friction-report-20261118.md for its original scope and retained first static failures.
## Fixed packed solution for native integration compatibility

pack-pipelines.json.gz is the complete I3w accepted payload, 26,596 bytes, SHA256 883dc7603c69d5789ea1e697ec8d64be33cb85c998bd244e4e05aba1a6b235a0. Original: artifacts/area-pressure-resident-friction-pack-20261118-01/payloads/test_two_resident_pipelines_wi0/entry-pack-pipelines.json.gz; product/test commit de7c31e3. It contains the original source fields, packed shared-DOF inputs, all raw solver outputs, and independent certificates.

I3x consumes the three actual FREE input lanes and the actual damped scalar lane, excluding the explicitly synthetic nonzero-JB lane. It aggregates stored full forces on HOST and invokes each original native integrator once; it never reruns the pressure/friction solver or advances a Scene. It verifies the stored forces at the actual newly integrated velocity and independently checks native q coordinates. The fresh clock is reconstructed as (0,0) only after proving original initial tick/time and input_time are zero; the original payload did not store raw clock words. This is not a new public force record, full live pipeline, or general proof of floating-point equivalence.

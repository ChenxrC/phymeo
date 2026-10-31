# S2-FB-G closed-payload storage ruling

Status: GRANTED_CLOSED_PAYLOAD_FILE_IO_ONLY. This supplements the existing sole HOST validation grant for the original S2 author. It does not create a second project execution slot or change any mathematical, source, test, resource or acceptance contract.

## Reason and exact scope

Run13 at tested source8fba8d091221c6a43520e5036da291b32e4f12f3 is still running. At15:44:30+08 the original author observed399 closed payloads and3406237696 free bytes; a conservative forecast for the remaining fixed grid left about598 MB before further logs/snapshots. Additional filesystem growth was outside the saved-payload estimate. No unrelated files were inspected or removed. At15:46:40 there were442 closed payloads and about3.274 GB free.

This permission applies ONLY to the1413 predeclared logical payload JSON files produced by this run under C:/Users/38347/AppData/Local/Temp/p65-shell-boundary-local-13bd742-serial. The existing predeclared membership SHA is b8eff403dedd622b43a3815a8ad646bb35ab1008e7a5ca86ac70c74dce1dc8d6. Logs, source/config snapshots, manifests, SHA sidecars, audit scripts, receipts, other runs and all unrelated files are excluded from removal.

The author checked the five selected tests as text: save writes/closes JSON before writing/closing its sidecar; selected tests do not subsequently read saved paths. Static and sweep rebuild comparisons use in-memory payload values. Endpoint and point tests likewise have no later file consumer. The only later reader is the separately scheduled saved audit. The owner must additionally keep each logical path's predeclared single-write identity; a file with any pending writer, reader or overwrite is ineligible.

## Per-file transaction

Use bounded streaming PowerShell/.NET file IO, owned by the SAME original author. It may process independently closed payloads while run13 continues because it performs only storage IO, not project import, numerical analysis, a geometry probe, test or saved-certificate audit. Do not start a second Python/project process or compress a file still being published. Process files sequentially in bounded batches and retain current capacity observations.

For every selected logical path:

1. Resolve and verify its absolute path remains under this run's intended payload root and belongs to the fixed membership. Require the original basename-matching sidecar, valid64-hex SHA and a completely closed, ordinary JSON file. Record its size and fresh streaming SHA; require sidecar equality. A missing, malformed, changing or ambiguous entry is left untouched and reported.
2. Create a NEW temporary gzip file beside it, using one independent gzip stream for the entire original byte sequence. Never overwrite an existing compressed file, follow an unexpected link or combine multiple payloads into one archive. Existing entries require checking their prior transaction record before reuse.
3. Close the writer. Independently reopen the gzip through a new decompression stream and recompute the entire original byte count and SHA. Require exact equality to the original and unchanged sidecar. Record compressed file size and SHA separately. Decompression verification is streaming; do not rehydrate another plain JSON on disk.
4. Recheck the original plain file's size and SHA immediately before removal. Publish the verified gzip under its final unique path and durably write a per-file transaction record and logical-path index binding original path, original size/SHA, sidecar path/SHA, compressed path/size/SHA, independent decompression size/SHA and transaction status. Preserve a recoverable original representation throughout.
5. ONLY after all preceding checks succeed may Remove-Item -LiteralPath remove the redundant original plain JSON. No recursive deletion, glob deletion or computed shell command is allowed. Record completion and verify the compressed representation and sidecar remain. If removal fails, retain both and record their exact equivalence; do not remove either representation based merely on an attempted operation.

No truncation, field omission, JSON reserialization, precision change, normalization or lossy encoding is allowed. A failed transaction cannot authorize deletion. Retain the transaction/error records and any necessary recoverable data. A pre-existing artifact is never overwritten to hide a failure.

## Later audit and parent archive

The post-run saved audit remains sequential AFTER the HOST run exits. Extend its byte reader to resolve the immutable logical path either to the plain file or to the independently verified indexed gzip. Validate compressed size/SHA and independently stream-decompress to the original logical size/SHA. A suffix alone is not proof of validity. Missing/extra logical members, unindexed compressed files, conflicting representations, corrupt streams or sidecar inconsistencies fail the audit.

The original1413 complete payloads and704 predeclared rebuild pairs are unchanged. Compare FULL decompressed bytes for each pair; no fields are removed or normalized. Preserve the full original input/claim/verification receipt and actual same-call counters. Do not rerun a producer, verifier, fixture or trajectory to reconstruct evidence. The observer may parse the original bytes for its already authorized assertions only after run completion.

The final manifest must distinguish physical stored files from logical original payloads and include every sidecar, index and transaction receipt. Parent later copies and verifies the stored compressed representation without recreating all plain JSON on the same disk. Original logical evidence is exactly recoverable; compression is storage management only and supplies no test or physical pass.

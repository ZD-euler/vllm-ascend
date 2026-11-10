No-audit ordinary-DDR qualification probe v2.

Use only the new SDK17 private build from the strict matching archive stager.
The MMC new config key defaults false. This probe explicitly sets it true;
it remains device_sdma and requests a real10GiB pool. MF binaries are unchanged.
No LD_AUDIT, LD_PRELOAD or DSA_MMC_DDR_NUMA is set in the child. There is no
NUMA promise; actual_NUMA/actual_2MiB_10GiB_placement_confirmed remain null.
No old audit allocation line is synthesized. The actual allocator decision is
qualified by reviewed SDK/MF source+CPU contracts, then real functional tests.
The library must contain the supported config key before any NPU admission.

Run in the existing validated A3 private namespace/environment, writable
private /tmp/log mounts already arranged by the root agent. Do not install
libraries or change global driver/VM settings. Do not run concurrently with
another owned probe/model. The root agent alone verifies idle die8 and prior
owned model+Meta exit before allowing the child to initialize the NPU.

python run_probe.py --sdk-site /ABS/sdk17-site --mf-site /ABS/old-sdk13/site \
    --unrestricted --output /ABS/NEW/sdk17-old-acquire-probe \
    --timeout-seconds 240 --deadline-unix AUTHORIZED_DEADLINE_UNIX

No --node and no --audit are allowed with --unrestricted.
The supervisor starts only a private CPU Meta and one CPU-preimported worker.
After public-import-ready.json, root rechecks actual pid/startticks/boot_id,
the physical8 resource scope and prior owned exit. Root writes admission.json
by copying ready's pid/startticks/boot_id/run_id/node (node:null), then adds:
physical_devices:[8], issued_unix:<fresh time.time()>,
prior_model_scope_empty_verified_by_parent:true. No prefilled admission.

Acceptance needs public-result.json status passed, requested pool10GiB,
worker actual init0, two real16MiB D2H/H2D byte-exact transfers (contiguous
and200 sparse descriptors), write_finish/readlease/remove0, unregister0,
close0, real worker returncode0 after natural interpreter shutdown and
captured owned scope empty. Any abort/doublefree/timeout/unknown rejects it.
The config/source policy does not imply SDMA/MTE cross-die accessibility.
Only after this process exits and owned scope is empty, start another fresh
root/process with the same SDK17/MF/config and --combined. This checks the
single AddWithInfo's KeyInfos against one independent flag0 query. No private
key, address or activation is exported. The extra query is correctness-only,
so do not use this probe as the production prepare-latency benchmark.

After both probe arms pass, the four-die6layer model still needs its real
MTE/SFA/LI oracle and natural teardown under this exact configuration. Use the
same SDK17/config for sourceV15(oldAdd+Get) and sourceV16(AddWithInfo) to isolate
the connector change. Neither probe nor new source is qualified in advance.

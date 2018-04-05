// lld: utilities for replacement

// dump all target data
void dumpTargetData(HostDataToTargetListTy *DataList) {
  LLD_DP("Target data:\n");
  int i = 0;
  for (auto &HT : *DataList) {
    LLD_DP("Entry %2d: Base=" DPxMOD ", Valid=%d, Time=%lu, Size=%" PRId64
        ", Type=0x%" PRIx64 "\n", i, DPxPTR(HT.HstPtrBegin), HT.IsValid, HT.TimeStamp,
        HT.HstPtrEnd - HT.HstPtrBegin, HT.MapType);
    i++;
  }
  LLD_DP("\n");
}

// lld: utilities for replacement

/// Data attributes for each data reference used in an OpenMP target region.
enum mem_map_type {
  MEM_MAPTYPE_DEV = 0,
  MEM_MAPTYPE_UVM,
  MEM_MAPTYPE_HOST,
  MEM_MAPTYPE_UNDECIDE
};

// get map type
inline mem_map_type getMemMapType(int64_t MapType) {
  bool IsUVM = MapType & OMP_TGT_MAPTYPE_UVM;
  bool IsHost = MapType & OMP_TGT_MAPTYPE_HOST;
  if (IsUVM & IsHost)
    return MEM_MAPTYPE_UNDECIDE;
  else if (IsUVM)
    return MEM_MAPTYPE_UVM;
  else if (IsHost)
    return MEM_MAPTYPE_HOST;
  else
    return MEM_MAPTYPE_DEV;
}

// dump all target data
void dumpTargetData(HostDataToTargetListTy *DataList) {
  LLD_DP("Target data:\n");
  int i = 0;
  for (auto &HT : *DataList) {
    LLD_DP("Entry %2d: Base=" DPxMOD ", Valid=%d, Deleted=%d, Reuse=%ld, Time=%lu, Size=%" PRId64
        ", Type=0x%" PRIx64 "\n", i, DPxPTR(HT.HstPtrBegin), HT.IsValid, HT.IsDeleted, HT.Reuse, HT.TimeStamp,
        HT.HstPtrEnd - HT.HstPtrBegin, HT.MapType);
    i++;
  }
  LLD_DP("\n");
}

// replace a data object
void placeDataObj(DeviceTy &Device, int32_t idx, int64_t &MapType, int64_t Size, uint64_t LTC, bool data_region) {
  if (data_region) {
    LLD_DP("  Arg %d mapping is not decided\n", idx);
    MapType |= OMP_TGT_MAPTYPE_UVM;
    MapType |= OMP_TGT_MAPTYPE_HOST;
  } else {
    unsigned LocalReuse = (MapType & OMP_TGT_MAPTYPE_LOCAL_REUSE) >> 20;
    double LocalReuseFloat = (double)LocalReuse / 8.0;
    double Density = LocalReuseFloat * LTC / Size;
    if (Density < 0.5) {
      LLD_DP("  Arg %d is intended for UM (%f)\n", idx, Density);
      MapType |= OMP_TGT_MAPTYPE_UVM;
    } else
      LLD_DP("  Arg %d is intended for device (%f)\n", idx, Density);
      MapType |= OMP_TGT_MAPTYPE_SDEV;
  }
}

// replace a data object
void replaceDataObj(DeviceTy &Device, int32_t idx, int64_t &MapType, int64_t Size, uint64_t LTC, bool data_region) {
  int64_t AvailSize = 0;
  int64_t Reuse = (MapType & OMP_TGT_MAPTYPE_RANK) >> 12;
  std::vector<HostDataToTargetTy*> DeleteList;
  for (auto &HT : Device.HostDataToTargetMap) {
    // find objects with poorer locality
    if (HT.Reuse > Reuse && (HT.MapType & OMP_TGT_MAPTYPE_HOST) != OMP_TGT_MAPTYPE_HOST) {
      int64_t HTSize = HT.HstPtrEnd - HT.HstPtrBegin;
      if (!HT.IsDeleted && HTSize > 256) { // Do not replace small objects
        AvailSize += HTSize;
        DeleteList.push_back(&HT);
        if (AvailSize >= Size)
          break;
      }
    }
  }

  if (AvailSize < Size) {
    LLD_DP("  No enough space for replacement (%lu)\n", AvailSize);
    if (getMemMapType(MapType) != MEM_MAPTYPE_HOST) {
      LLD_DP("  Arg %d is mapped to host\n", idx);
      MapType |= OMP_TGT_MAPTYPE_HOST;
    }
    return;
  }

  // release space
  for (auto *E : DeleteList) {
    int64_t ESize = E->HstPtrEnd - E->HstPtrBegin;
    bool IsUM = (E->HstPtrBegin == E->TgtPtrBegin);
    // Retrieve data to host
    if (!IsUM) {
      LLD_DP("  Replace " DPxMOD " from device (" DPxMOD "), size=%ld\n", DPxPTR(E->HstPtrBegin), DPxPTR(E->TgtPtrBegin), ESize);
      if (E->IsValid) {
        int rt = Device.data_retrieve((void*)E->HstPtrBegin, (void*)E->TgtPtrBegin, ESize);
        if (rt != OFFLOAD_SUCCESS) {
          LLD_DP("  Error: Copying data from device failed.\n");
        }
        E->IsValid = false;
      }
      Device.deviceSize -= ESize;
      Device.RTL->data_delete(Device.RTLDeviceID, (void *)E->TgtPtrBegin);
      E->IsDeleted = true;
    } else {
      LLD_DP("  Replace " DPxMOD " from UM, size=%ld\n", DPxPTR(E->HstPtrBegin), ESize);
      // Retrieve to host
      Device.RTL->data_opt(Device.RTLDeviceID, ESize, (void *)E->HstPtrBegin, 5);
      // Pin to host
      Device.RTL->data_opt(Device.RTLDeviceID, ESize, (void *)E->HstPtrBegin, 0);
      Device.umSize -= ESize;
      E->IsValid = false;
    }
  }
  // place data
  placeDataObj(Device, idx, MapType, Size, LTC, data_region);
}

// lld: compare rank
bool compareRank(std::pair<int32_t, int64_t> A, std::pair<int32_t, int64_t> B) {
  return (A.second < B.second);
}

// lld: decide mapping based on rank
std::pair<int64_t*, int64_t*> target_uvm_data_mapping_opt(DeviceTy &Device, void **args_base, void **args, int32_t arg_num, int64_t *arg_sizes, int64_t *arg_types, bool data_region) {
  int64_t used_dev_size = 0;
  uint64_t ltc = Device.loopTripCnt;
  LLD_DP("%s\t(#iter: %lu    device: %lu    UM: %lu)\n", (data_region ? "DATA\t" : "COMPUTE"), ltc, Device.deviceSize, Device.umSize);
  GlobalTimeStamp++;
  //dumpTargetData(&Device.HostDataToTargetMap);
  std::vector<std::pair<int32_t, int64_t>> argList;
  int64_t *new_arg_types = (int64_t*)malloc(sizeof(int64_t)*arg_num);
  int64_t *new_arg_sizes = (int64_t*)malloc(sizeof(int64_t)*arg_num);
  for (int32_t i = 0; i < arg_num; ++i) {
    new_arg_types[i] = arg_types[i];
    new_arg_sizes[i] = arg_sizes[i];
    //if (arg_types[i] & OMP_TGT_MAPTYPE_IMPLICIT)
    //  continue;
    if (!(arg_types[i] & OMP_TGT_MAPTYPE_RANK))
      continue;
    unsigned Rank = (arg_types[i] & OMP_TGT_MAPTYPE_RANK) >> 12;
    argList.push_back(std::make_pair(i, Rank));
  }
  std::sort(argList.begin(), argList.end(), compareRank);
  for (auto I : argList) {
    int32_t idx = I.first;
    if (GMode == 1) // UM mode
      new_arg_types[idx] |= OMP_TGT_MAPTYPE_UVM;
    else if (GMode == 2) { // DEV mode
    } else if (GMode == 3) // HOST mode
      new_arg_types[idx] |= OMP_TGT_MAPTYPE_HOST;
    else if (GMode == 4) // HYB mode
      new_arg_types[idx] |= OMP_TGT_MAPTYPE_HYB;
    else { // Normal processing
      uint64_t DataSize = arg_sizes[idx];
      LookupResult lr = Device.lookupMapping(args_base[idx], DataSize);
      if (lr.Flags.IsContained || lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter)
        DataSize = lr.Entry->HstPtrEnd - lr.Entry->HstPtrBegin;
      if (DataSize == 0)
        continue;
      if (lr.Entry != Device.HostDataToTargetMap.end() &&
          (!lr.Entry->Decided || !lr.Entry->IsValid)) {
        // restore recorded maptype
        new_arg_types[idx] &= ~0x3ff;
        new_arg_types[idx] |= lr.Entry->MapType & 0x3ff;
        // restore size
        new_arg_sizes[idx] = DataSize;
      }
      if (lr.Entry != Device.HostDataToTargetMap.end() &&
          ((lr.Entry->MapType & OMP_TGT_MAPTYPE_HOST) != OMP_TGT_MAPTYPE_HOST)) {
        continue;
      } else if (used_dev_size + Device.deviceSize + Device.umSize + DataSize < total_dev_size) {
        placeDataObj(Device, idx, new_arg_types[idx], DataSize, ltc, data_region);
        used_dev_size += DataSize;
      } else
        replaceDataObj(Device, idx, new_arg_types[idx], DataSize, ltc, data_region);
      LLD_DP("  Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
          ", Type=0x%" PRIx64 "\n", idx, DPxPTR(args_base[idx]), DPxPTR(args[idx]),
          new_arg_sizes[idx], new_arg_types[idx]);
    }
  }
  return std::make_pair(new_arg_types, new_arg_sizes);
}

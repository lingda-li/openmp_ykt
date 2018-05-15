// lld: utilities for replacement

/// Data object cluster
enum data_cluster_type {
  CLUSTER_MAPTYPE_DEV = 0,
  CLUSTER_MAPTYPE_MIX,
  CLUSTER_MAPTYPE_HST
};

struct DataClusterTy {
  void *BasePtr;
  std::list<HostDataToTargetTy*> Members;
  data_cluster_type Type;
  uint64_t Size = 0;
  double Priority = 0.0;

  DataClusterTy(void *Base) : BasePtr(Base) {}
};

/// Data attributes for each data reference used in an OpenMP target region.
enum mem_map_type {
  MEM_MAPTYPE_DEV = 0,
  MEM_MAPTYPE_SDEV,
  MEM_MAPTYPE_UVM,
  MEM_MAPTYPE_HOST,
  MEM_MAPTYPE_UNDECIDE
};

// get map type
inline mem_map_type getMemMapType(int64_t MapType) {
  bool IsUVM = MapType & OMP_TGT_MAPTYPE_UVM;
  bool IsHost = MapType & OMP_TGT_MAPTYPE_HOST;
  bool SoftDev = MapType & OMP_TGT_MAPTYPE_SDEV;
  if (IsUVM & IsHost)
    return MEM_MAPTYPE_UNDECIDE;
  else if (IsUVM)
    return MEM_MAPTYPE_UVM;
  else if (IsHost)
    return MEM_MAPTYPE_HOST;
  else if (SoftDev)
    return MEM_MAPTYPE_SDEV;
  else
    return MEM_MAPTYPE_DEV;
}

inline void setMemMapType(int64_t &MapType, mem_map_type Type) {
  MapType &= ~(OMP_TGT_MAPTYPE_UVM | OMP_TGT_MAPTYPE_HOST | OMP_TGT_MAPTYPE_SDEV);
  if (MEM_MAPTYPE_UNDECIDE)
    MapType |= (OMP_TGT_MAPTYPE_UVM | OMP_TGT_MAPTYPE_HOST);
  else if (MEM_MAPTYPE_UVM)
    MapType |= OMP_TGT_MAPTYPE_UVM;
  else if (MEM_MAPTYPE_HOST)
    MapType |= OMP_TGT_MAPTYPE_HOST;
  else if (MEM_MAPTYPE_SDEV)
    MapType |= OMP_TGT_MAPTYPE_SDEV;
}

bool isInDevCluster(HostDataToTargetTy *E) {
  for (auto *C : E->Clusters) {
    if (C->Type == CLUSTER_MAPTYPE_DEV)
      return true;
  }
  return false;
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
void placeDataObj(DeviceTy &Device, HostDataToTargetTy *HT, int32_t idx, int64_t &MapType, int64_t Size, void *Base, uint64_t LTC, bool data_region) {
  if (data_region) {
    LLD_DP("  Arg %d (" DPxMOD ") mapping is not decided\n", idx, DPxPTR(Base));
    MapType |= OMP_TGT_MAPTYPE_UVM;
    MapType |= OMP_TGT_MAPTYPE_HOST;
  } else {
    unsigned LocalReuse = (MapType & OMP_TGT_MAPTYPE_LOCAL_REUSE) >> 20;
    double LocalReuseFloat = (double)LocalReuse / 8.0;
    double Density = LocalReuseFloat * LTC / Size;
    if (Density < 0.5) {
      LLD_DP("  Arg %d (" DPxMOD ") is intended for UM (%f)\n", idx, DPxPTR(Base), Density);
      MapType |= OMP_TGT_MAPTYPE_UVM;
    } else
      LLD_DP("  Arg %d (" DPxMOD ") is intended for device (%f)\n", idx, DPxPTR(Base), Density);
      MapType |= OMP_TGT_MAPTYPE_SDEV;
  }
  if (HT)
    HT->ChangeMap = true;
}

// release space
void releaseDataObj(DeviceTy &Device, HostDataToTargetTy *E) {
  int64_t Size = E->HstPtrEnd - E->HstPtrBegin;
  mem_map_type PreMap = getMemMapType(E->MapType);
  assert(PreMap != MEM_MAPTYPE_UNDECIDE);
  if (PreMap == MEM_MAPTYPE_DEV) {
    LLD_DP("  Replace " DPxMOD " from device (" DPxMOD "), size=%ld\n", DPxPTR(E->HstPtrBegin), DPxPTR(E->TgtPtrBegin), Size);
    if (E->IsValid) {
      int rt = Device.data_retrieve((void*)E->HstPtrBegin, (void*)E->TgtPtrBegin, Size);
      if (rt != OFFLOAD_SUCCESS)
        LLD_DP("  Error: Copying data from device failed.\n");
      E->IsValid = false;
    }
    Device.deviceSize -= Size;
    Device.RTL->data_delete(Device.RTLDeviceID, (void *)E->TgtPtrBegin);
    E->IsDeleted = true;
  } else if (PreMap == MEM_MAPTYPE_UVM) {
    LLD_DP("  Replace " DPxMOD " from UM, size=%ld\n", DPxPTR(E->HstPtrBegin), Size);
    Device.RTL->data_opt(Device.RTLDeviceID, Size, (void *)E->HstPtrBegin, 0); // pin to host
    Device.RTL->data_opt(Device.RTLDeviceID, Size, (void *)E->HstPtrBegin, 5); // prefetch to host
    Device.umSize -= Size;
    //E->IsValid = false;
    //E->IsDeleted = true;
    setMemMapType(E->MapType, MEM_MAPTYPE_HOST);
  } else if (PreMap == MEM_MAPTYPE_HOST) {
    LLD_DP("  Error: Try to replace a host object.\n");
  } else if (PreMap == MEM_MAPTYPE_SDEV) {
    LLD_DP("  Replace " DPxMOD " from soft device, size=%ld\n", DPxPTR(E->HstPtrBegin), Size);
    Device.RTL->data_opt(Device.RTLDeviceID, Size, (void *)E->HstPtrBegin, 0); // pin to host
    Device.RTL->data_opt(Device.RTLDeviceID, Size, (void *)E->HstPtrBegin, 5); // prefetch to host
    Device.deviceSize -= Size;
    setMemMapType(E->MapType, MEM_MAPTYPE_HOST);
  }
  return;
  /*
  */

  bool IsUM = (E->HstPtrBegin == E->TgtPtrBegin);
  // Retrieve data to host
  if (!IsUM) {
    LLD_DP("  Replace " DPxMOD " from device (" DPxMOD "), size=%ld\n", DPxPTR(E->HstPtrBegin), DPxPTR(E->TgtPtrBegin), Size);
    if (E->IsValid) {
      int rt = Device.data_retrieve((void*)E->HstPtrBegin, (void*)E->TgtPtrBegin, Size);
      if (rt != OFFLOAD_SUCCESS) {
        LLD_DP("  Error: Copying data from device failed.\n");
      }
      E->IsValid = false;
    }
    Device.deviceSize -= Size;
    Device.RTL->data_delete(Device.RTLDeviceID, (void *)E->TgtPtrBegin);
    E->IsDeleted = true;
    //FIXME: also pin to host instead?
  } else {
    LLD_DP("  Replace " DPxMOD " from UM, size=%ld\n", DPxPTR(E->HstPtrBegin), Size);
    // Pin to host
    Device.RTL->data_opt(Device.RTLDeviceID, Size, (void *)E->HstPtrBegin, 0);
    // Retrieve to host
    Device.RTL->data_opt(Device.RTLDeviceID, Size, (void *)E->HstPtrBegin, 5);
    //Device.umSize -= Size;
    E->IsValid = false;
    E->IsDeleted = true;
    if (PreMap == MEM_MAPTYPE_UVM)
      Device.umSize -= Size;
    else
      Device.deviceSize -= Size;
  }
}

// replace a data object
bool replaceDataObj(DeviceTy &Device, HostDataToTargetTy *HT, int32_t idx, int64_t &MapType, int64_t Size, int64_t AvailSize, void *Base, uint64_t LTC, bool data_region) {
  int64_t Reuse = (MapType & OMP_TGT_MAPTYPE_RANK) >> 12;
  std::vector<HostDataToTargetTy*> DeleteList;
  for (auto &HT : Device.HostDataToTargetMap) {
    // if it belongs to the current cluster
    if (HT.Irreplaceable)
      continue;
    // if it belongs to a pinned cluster
    if (isInDevCluster(&HT)) {
      continue;
    }
    // find objects with poorer locality
    if (HT.Reuse > Reuse && (HT.MapType & OMP_TGT_MAPTYPE_HOST) != OMP_TGT_MAPTYPE_HOST) {
      int64_t HTSize = HT.HstPtrEnd - HT.HstPtrBegin;
      if (!HT.IsDeleted && HTSize > 256) { // Do not replace small objects
        AvailSize += HTSize;
        DeleteList.push_back(&HT);
        // FIXME: not those with worst locality
        if (AvailSize >= Size)
          break;
      }
    }
  }

  if (AvailSize < Size) {
    LLD_DP("  No enough space for replacement (%lu)\n", AvailSize);
    if (getMemMapType(MapType) != MEM_MAPTYPE_HOST) {
      LLD_DP("  Arg %d (" DPxMOD ") is mapped to host\n", idx, DPxPTR(Base));
      MapType |= OMP_TGT_MAPTYPE_HOST;
    }
    return false;
  }

  // release space
  for (auto *E : DeleteList)
    releaseDataObj(Device, E);
  // place data
  placeDataObj(Device, HT, idx, MapType, Size, Base, LTC, data_region);
  return true;
}

DataClusterTy *DeviceTy::lookupCluster(void *Base) {
  DataClusterListTy::iterator it;
  for (it = DataClusters.begin();
      it != DataClusters.end(); ++it) {
    if (it->BasePtr == Base)
      return &(*it);
  }
  return NULL;
}

void dumpClusters(DataClusterListTy *CList) {
  LLD_DP("Cluster:\n");
  int i = 0;
  for (auto &HT : *CList) {
    LLD_DP("Cluster %2d: Base=" DPxMOD ", Type=%d\n", i, DPxPTR(HT.BasePtr), HT.Type);
    i++;
  }
  LLD_DP("\n");
}

void cleanReplaceMetadata(DeviceTy &Device) {
  for (auto &HT : Device.HostDataToTargetMap)
    HT.Irreplaceable = false;
}

void replaceDataCluster(DeviceTy &Device, int64_t Size, int64_t AvailSize, std::vector<std::pair<int32_t, int64_t>> &argList, LookupResult *LRs, int64_t *argTypes, int64_t *argSizes, void **argBases, uint64_t LTC) {
  int64_t OriAvailSize = AvailSize;
  int64_t used_dev_size = 0;
  if (AvailSize >= Size) {
    LLD_DP("  Cluster " DPxMOD " uses device mapping\n", DPxPTR(Device.CurrentCluster->BasePtr));
    Device.CurrentCluster->Type = CLUSTER_MAPTYPE_DEV;
    for (auto I : argList) {
      int32_t idx = I.first;
      LookupResult lr = LRs[idx];
      if (lr.Entry != Device.HostDataToTargetMap.end() && lr.Entry->Irreplaceable)
        continue;
      placeDataObj(Device, &(*lr.Entry), idx, argTypes[idx], argSizes[idx], argBases[idx], LTC, false);
    }
    cleanReplaceMetadata(Device);
    return;
  }

  std::vector<HostDataToTargetTy*> ReplaceList;
  for (auto &HT : Device.HostDataToTargetMap) {
    // if it belongs to the current cluster
    if (HT.Irreplaceable)
      continue;
    // if it belongs to a pinned cluster
    if (isInDevCluster(&HT)) {
      continue;
    }
    // find objects with poorer locality
    if ((HT.MapType & OMP_TGT_MAPTYPE_HOST) != OMP_TGT_MAPTYPE_HOST) {
      int64_t HTSize = HT.HstPtrEnd - HT.HstPtrBegin;
      if (!HT.IsDeleted && HTSize > 256) { // Do not replace small objects
        AvailSize += HTSize;
        ReplaceList.push_back(&HT);
      }
    }
  }

  if (AvailSize < Size) {
    LLD_DP("  Cluster " DPxMOD " uses mixed mapping\n", DPxPTR(Device.CurrentCluster->BasePtr));
    Device.CurrentCluster->Type = CLUSTER_MAPTYPE_MIX;
    for (auto I : argList) {
      int32_t idx = I.first;
      LookupResult lr = LRs[idx];
      if (lr.Entry != Device.HostDataToTargetMap.end() && lr.Entry->Irreplaceable)
        continue;
      if (argSizes[idx] <= (int64_t)(total_dev_size - used_dev_size - Device.deviceSize - Device.umSize)) {
        placeDataObj(Device, &(*lr.Entry), idx, argTypes[idx], argSizes[idx], argBases[idx], LTC, false);
        used_dev_size += argSizes[idx];
      } else if (replaceDataObj(Device, &(*lr.Entry), idx, argTypes[idx], argSizes[idx], total_dev_size - used_dev_size - Device.deviceSize - Device.umSize, argBases[idx], LTC, false))
        used_dev_size += argSizes[idx];
    }
  } else {
    LLD_DP("  Cluster " DPxMOD " uses device mapping\n", DPxPTR(Device.CurrentCluster->BasePtr));
    Device.CurrentCluster->Type = CLUSTER_MAPTYPE_DEV;
    AvailSize = OriAvailSize;
    for (auto *E : ReplaceList) {
      releaseDataObj(Device, E);
      AvailSize += E->HstPtrEnd - E->HstPtrBegin;
      if (AvailSize >= Size)
        break;
    }
    for (auto I : argList) {
      int32_t idx = I.first;
      LookupResult lr = LRs[idx];
      if (lr.Entry != Device.HostDataToTargetMap.end() && lr.Entry->Irreplaceable)
        continue;
      placeDataObj(Device, &(*lr.Entry), idx, argTypes[idx], argSizes[idx], argBases[idx], LTC, false);
    }
  }
  cleanReplaceMetadata(Device);
}

// Used by target_data_begin
// Return the target pointer begin (where the data will be moved).
// Allocate memory if this is the first occurrence if this mapping.
// Increment the reference counter.
// If NULL is returned, then either data allocation failed or the user tried
// to do an illegal mapping.
void *DeviceTy::getOrAllocTgtPtr(void *HstPtrBegin, void *HstPtrBase,
    //int64_t Size, bool &IsNew, bool IsImplicit, bool UpdateRefCount) {
    // lld: uvm
    int64_t Size, bool &IsNew, bool IsImplicit, bool UpdateRefCount,
    int64_t MapType) {
  // lld: get mapping types
  bool UVM = MapType & OMP_TGT_MAPTYPE_UVM;
  bool PinHost = MapType & OMP_TGT_MAPTYPE_HOST;
  bool HYB = MapType & OMP_TGT_MAPTYPE_HYB;
  bool SoftDev = MapType & OMP_TGT_MAPTYPE_SDEV;

  void *rc = NULL;
  DataMapMtx.lock();
  LookupResult lr = lookupMapping(HstPtrBegin, Size);
  auto *DMEP = &(*lr.Entry);

  // Check if the pointer is contained.
  if (lr.Flags.IsContained ||
      ((lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter) && IsImplicit)) {
    auto &HT = *lr.Entry;
    IsNew = false;

    if (UpdateRefCount)
      ++HT.RefCount;

    // lld: delay decision
    if (!HT.Decided) {
      if (UVM && PinHost) { // delay decision
      } else if (UVM) {
        LLD_DP("  Map " DPxMOD " to UM, size=%ld\n", DPxPTR(HstPtrBegin), Size);
        HT.TgtPtrBegin = (uintptr_t)HstPtrBegin;
        umSize += Size;
        HT.Decided = true;
        HT.MapType = MapType;
        //RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 2);
        IsNew = true;
      } else if (SoftDev) {
        LLD_DP("  Map " DPxMOD " to soft device, size=%ld\n", DPxPTR(HstPtrBegin), Size);
        HT.TgtPtrBegin = (uintptr_t)HstPtrBegin;
        RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 4); // pin to device
        RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 1); // prefetch
        deviceSize += Size;
        HT.Decided = true;
        HT.MapType = MapType;
        IsNew = true;
      } else if (PinHost) {
        HT.TgtPtrBegin = (uintptr_t)HstPtrBegin;
        RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 0);
        HT.Decided = true;
        HT.MapType = MapType;
        IsNew = true;
      } else {
        HT.TgtPtrBegin = (uintptr_t)RTL->data_alloc(RTLDeviceID, Size, HstPtrBegin);
        deviceSize += Size;
        HT.Decided = true;
        HT.MapType = MapType;
        IsNew = true;
        LLD_DP("  Map " DPxMOD " to device (" DPxMOD "), size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size);
      }
    } else if (HT.ChangeMap) {
      mem_map_type PreMap = getMemMapType(HT.MapType);
      assert(PreMap != MEM_MAPTYPE_UNDECIDE);
      if (UVM && PinHost) { // delay decision
        LLD_DP("Error:  " DPxMOD " becomes undecided after mapped, size=%ld\n", DPxPTR(HstPtrBegin), Size);
      } else if (UVM) {
        if (PreMap == MEM_MAPTYPE_DEV) {
          assert(HT.TgtPtrBegin != HT.HstPtrBegin);
          LLD_DP("  Remap " DPxMOD " from device (" DPxMOD ") to UM, size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size);
          deviceSize -= Size;
          RTL->data_delete(RTLDeviceID, (void *)HT.TgtPtrBegin);
          HT.TgtPtrBegin = (uintptr_t)HstPtrBegin;
          umSize += Size;
        } else if (PreMap == MEM_MAPTYPE_UVM) {
          // do nothing
        } else if (PreMap == MEM_MAPTYPE_HOST) {
          LLD_DP("  Remap " DPxMOD " from host to UM, size=%ld\n", DPxPTR(HstPtrBegin), Size);
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 6); // unpin
          umSize += Size;
        } else if (PreMap == MEM_MAPTYPE_SDEV) {
          LLD_DP("  Remap " DPxMOD " from soft device to UM, size=%ld\n", DPxPTR(HstPtrBegin), Size);
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 6); // unpin
          deviceSize -= Size;
          umSize += Size;
        }
      } else if (SoftDev) {
        if (PreMap == MEM_MAPTYPE_DEV) {
          assert(HT.TgtPtrBegin != HT.HstPtrBegin);
          LLD_DP("  Remap " DPxMOD " from device (" DPxMOD ") to soft device, size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size);
          RTL->data_delete(RTLDeviceID, (void *)HT.TgtPtrBegin);
          HT.TgtPtrBegin = (uintptr_t)HstPtrBegin;
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 4); // pin to device
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 1); // prefetch
        } else if (PreMap == MEM_MAPTYPE_UVM) {
          LLD_DP("  Remap " DPxMOD " from UM to soft device, size=%ld\n", DPxPTR(HstPtrBegin), Size);
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 4); // pin to device
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 1); // prefetch
          umSize -= Size;
          deviceSize += Size;
        } else if (PreMap == MEM_MAPTYPE_HOST) {
          LLD_DP("  Remap " DPxMOD " from host to soft device, size=%ld\n", DPxPTR(HstPtrBegin), Size);
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 4); // pin to device
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 1); // prefetch
          deviceSize += Size;
        } else if (PreMap == MEM_MAPTYPE_SDEV) {
          // do nothing
        }
      } else if (PinHost) {
        if (PreMap == MEM_MAPTYPE_DEV) {
          assert(HT.TgtPtrBegin != HT.HstPtrBegin);
          deviceSize -= Size;
          RTL->data_delete(RTLDeviceID, (void *)HT.TgtPtrBegin);
          LLD_DP("  Remap " DPxMOD " from device (" DPxMOD ") to host, size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size);
          HT.TgtPtrBegin = (uintptr_t)HstPtrBegin;
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 0);
        } else if (PreMap == MEM_MAPTYPE_UVM) {
          LLD_DP("  Remap " DPxMOD " from UM to host, size=%ld\n", DPxPTR(HstPtrBegin), Size);
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 0); // pin to host
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 5); // prefetch to host
          umSize -= Size;
        } else if (PreMap == MEM_MAPTYPE_HOST) {
          // do nothing
        } else if (PreMap == MEM_MAPTYPE_SDEV) {
          LLD_DP("  Remap " DPxMOD " from soft device to host, size=%ld\n", DPxPTR(HstPtrBegin), Size);
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 0); // pin to host
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 5); // prefetch to host
          deviceSize -= Size;
        }
      } else {
        if (PreMap == MEM_MAPTYPE_DEV) {
          assert(HT.TgtPtrBegin != HT.HstPtrBegin);
          // do nothing
        } else if (PreMap == MEM_MAPTYPE_UVM) {
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 1); // prefetch to device
          HT.TgtPtrBegin = (uintptr_t)RTL->data_alloc(RTLDeviceID, Size, HstPtrBegin);
          int rt = RTL->data_submit(RTLDeviceID, (void*)HT.TgtPtrBegin, HstPtrBegin, Size);
          if (rt != OFFLOAD_SUCCESS)
            LLD_DP("Copying data to device failed.\n");
          LLD_DP("  Remap " DPxMOD " from UM to device (" DPxMOD "), size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size);
          deviceSize += Size;
          umSize -= Size;
        } else if (PreMap == MEM_MAPTYPE_HOST) {
          HT.TgtPtrBegin = (uintptr_t)RTL->data_alloc(RTLDeviceID, Size, HstPtrBegin);
          int rt = RTL->data_submit(RTLDeviceID, (void*)HT.TgtPtrBegin, HstPtrBegin, Size);
          if (rt != OFFLOAD_SUCCESS)
            LLD_DP("Copying data to device failed.\n");
          LLD_DP("  Remap " DPxMOD " from host to device (" DPxMOD "), size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size);
          deviceSize += Size;
        } else if (PreMap == MEM_MAPTYPE_SDEV) {
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 0); // pin to host
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 5); // prefetch to host
          HT.TgtPtrBegin = (uintptr_t)RTL->data_alloc(RTLDeviceID, Size, HstPtrBegin);
          int rt = RTL->data_submit(RTLDeviceID, (void*)HT.TgtPtrBegin, HstPtrBegin, Size);
          if (rt != OFFLOAD_SUCCESS)
            LLD_DP("Copying data to device failed.\n");
          LLD_DP("  Remap " DPxMOD " from soft device to device (" DPxMOD "), size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size);
        }
      }
      HT.MapType = MapType;
    }
    HT.ChangeMap = false;
    uintptr_t tp = HT.TgtPtrBegin + ((uintptr_t)HstPtrBegin - HT.HstPtrBegin);
    DP("Mapping exists%s with HstPtrBegin=" DPxMOD ", TgtPtrBegin=" DPxMOD ", "
        "Size=%ld,%s RefCount=%s\n", (IsImplicit ? " (implicit)" : ""),
        DPxPTR(HstPtrBegin), DPxPTR(tp), Size,
        (UpdateRefCount ? " updated" : ""),
        (CONSIDERED_INF(HT.RefCount)) ? "INF" :
            std::to_string(HT.RefCount).c_str());
    rc = (void *)tp;
    // lld: update replacement info
    HT.TimeStamp = GlobalTimeStamp;
  } else if ((lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter) && !IsImplicit) {
    // Explicit extension of mapped data - not allowed.
    DP("Explicit extension of mapping is not allowed.\n");
  } else if (lr.Flags.InvalidContained ||
      ((lr.Flags.InvalidExtendsB || lr.Flags.InvalidExtendsA) && IsImplicit)) { // lld: invalid
    auto &HT = *lr.Entry;
    IsNew = true;
    assert(HT.Decided);

    if (UpdateRefCount)
      ++HT.RefCount;

    HstPtrBegin = (void*)HT.HstPtrBegin;
    Size = HT.HstPtrEnd - HT.HstPtrBegin;
    uintptr_t tp;
    if (UVM && PinHost) { // delay decision
      tp = (uintptr_t)HstPtrBegin;
      HT.Decided = false;
      IsNew = false;
    } else if (UVM) {
      if (HT.TgtPtrBegin != HT.HstPtrBegin) {
        deviceSize -= Size;
        RTL->data_delete(RTLDeviceID, (void *)HT.TgtPtrBegin);
        LLD_DP("  Unmap " DPxMOD " from device (" DPxMOD "), size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size);
      }
      LLD_DP("  Remap " DPxMOD " to UM, size=%ld\n", DPxPTR(HstPtrBegin), Size);
      tp = (uintptr_t)HstPtrBegin;
      umSize += Size;
      //RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 2);
    } else if (SoftDev) {
      if (HT.TgtPtrBegin != HT.HstPtrBegin) {
        deviceSize -= Size;
        RTL->data_delete(RTLDeviceID, (void *)HT.TgtPtrBegin);
        LLD_DP("  Unmap " DPxMOD " from device (" DPxMOD "), size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size);
      }
      LLD_DP("  Remap " DPxMOD " to soft device, size=%ld\n", DPxPTR(HstPtrBegin), Size);
      tp = (uintptr_t)HstPtrBegin;
      RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 4); // pin to device
      RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 1); // prefetch
      deviceSize += Size;
    } else if (PinHost) {
      if (HT.TgtPtrBegin != HT.HstPtrBegin) {
        deviceSize -= Size;
        RTL->data_delete(RTLDeviceID, (void *)HT.TgtPtrBegin);
        LLD_DP("  Unmap " DPxMOD " from device (" DPxMOD "), size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size);
      }
      tp = (uintptr_t)HstPtrBegin;
      RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 0);
    } else {
      if (HT.TgtPtrBegin != HT.HstPtrBegin && !HT.IsDeleted) {
        LLD_DP("  Reassociate " DPxMOD " to device (" DPxMOD "), size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size);
        tp = HT.TgtPtrBegin;
      } else {
        LLD_DP("  Remap " DPxMOD " to device (" DPxMOD "), size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(tp), Size);
        tp = (uintptr_t)RTL->data_alloc(RTLDeviceID, Size, HstPtrBegin);
        deviceSize += Size;
      }
    }
    HT.IsValid = true;
    HT.TgtPtrBegin = tp;
    HT.MapType = MapType;
    HT.ChangeMap = false;
    // lld: replacement info
    HT.TimeStamp = GlobalTimeStamp;
    rc = (void *)tp;
  } else if ((lr.Flags.InvalidExtendsB || lr.Flags.InvalidExtendsA) && !IsImplicit) { // lld: invalid
    // FIXME: reallocate space if necessary
    // Explicit extension of mapped data - not allowed.
    LLD_DP("Explicit extension of mapping is not allowed.\n");
  } else if (Size) {
    // If it is not contained and Size > 0 we should create a new entry for it.
    IsNew = true;
    //uintptr_t tp = (uintptr_t)RTL->data_alloc(RTLDeviceID, Size, HstPtrBegin);
    // lld: uvm
    uintptr_t tp;
    if (UVM && PinHost) { // delay decision
      tp = (uintptr_t)HstPtrBegin;
      IsNew = false;
    } else if (UVM) {
      LLD_DP("  Map " DPxMOD " to UM, size=%ld\n", DPxPTR(HstPtrBegin), Size);
      tp = (uintptr_t)HstPtrBegin;
      umSize += Size;
      //RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 2);
    } else if (SoftDev) {
      LLD_DP("  Map " DPxMOD " to soft device, size=%ld\n", DPxPTR(HstPtrBegin), Size);
      tp = (uintptr_t)HstPtrBegin;
      RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 4); // pin to device
      RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 1); // prefetch
      deviceSize += Size;
    } else if (PinHost) {
      tp = (uintptr_t)HstPtrBegin;
      RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 0);
    } else if (HYB) {
      int64_t DevSize = Size * devMemRatio;
      if (DevSize < Size) {
        LLD_DP("  Map " DPxMOD " to both locations, size=%ld (%ld)\n", DPxPTR(HstPtrBegin), Size, DevSize);
        tp = (uintptr_t)HstPtrBegin;
        RTL->data_opt(RTLDeviceID, DevSize, HstPtrBegin, 4); // pin to device
        RTL->data_opt(RTLDeviceID, DevSize, HstPtrBegin, 1); // prefetch
        umSize += DevSize;
        RTL->data_opt(RTLDeviceID, Size-DevSize, HstPtrBegin+DevSize, 0); // pin to host
      } else {
        tp = (uintptr_t)RTL->data_alloc(RTLDeviceID, Size, HstPtrBegin);
        deviceSize += Size;
        LLD_DP("  Map " DPxMOD " to device (" DPxMOD "), size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(tp), Size);
      }
    } else {
      tp = (uintptr_t)RTL->data_alloc(RTLDeviceID, Size, HstPtrBegin);
      deviceSize += Size;
      LLD_DP("  Map " DPxMOD " to device (" DPxMOD "), size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(tp), Size);
    }
    DP("Creating new map entry: HstBase=" DPxMOD ", HstBegin=" DPxMOD ", "
        "HstEnd=" DPxMOD ", TgtBegin=" DPxMOD "\n", DPxPTR(HstPtrBase),
        DPxPTR(HstPtrBegin), DPxPTR((uintptr_t)HstPtrBegin + Size), DPxPTR(tp));
    HostDataToTargetTy DataEntry = HostDataToTargetTy((uintptr_t)HstPtrBase,
        (uintptr_t)HstPtrBegin, (uintptr_t)HstPtrBegin + Size, tp);
    // lld: delay decision
    if (UVM && PinHost)
      DataEntry.Decided = false;
    DataEntry.MapType = MapType;
    // lld: replacement info
    DataEntry.Reuse = (MapType & OMP_TGT_MAPTYPE_RANK) >> 12;
    DataEntry.TimeStamp = GlobalTimeStamp;
    DMEP = &DataEntry;
    HostDataToTargetMap.push_front(DataEntry);
    rc = (void *)tp;
  }

  DataMapMtx.unlock();
  // lld: insert to cluster
  if (CurrentCluster && IsNewCluster) {
    CurrentCluster->Members.push_front(DMEP);
    DMEP->Clusters.push_front(CurrentCluster);
  }
  return rc;
}

// lld: compare rank
bool compareRank(std::pair<int32_t, int64_t> A, std::pair<int32_t, int64_t> B) {
  return (A.second < B.second);
}

// lld: decide mapping based on rank
std::pair<int64_t*, int64_t*> target_uvm_data_mapping_opt(DeviceTy &Device, void **args_base, void **args, int32_t arg_num, int64_t *arg_sizes, int64_t *arg_types, void *host_ptr) {
  int64_t used_dev_size = 0;
  uint64_t ltc = Device.loopTripCnt;
  bool data_region = (host_ptr == NULL ? true : false);
  if (data_region)
    LLD_DP("DATA\t\t\t\t(#iter: %lu    device: %lu    UM: %lu)\n", ltc, Device.deviceSize, Device.umSize)
  else
    LLD_DP("COMPUTE (" DPxMOD ")\t(#iter: %lu    device: %lu    UM: %lu)\n", DPxPTR(host_ptr), ltc, Device.deviceSize, Device.umSize)
  GlobalTimeStamp++;
  //dumpTargetData(&Device.HostDataToTargetMap);

  double CP = 0.0;
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
    if (GMode == 1) // UM mode
      new_arg_types[i] |= OMP_TGT_MAPTYPE_UVM;
    else if (GMode == 2) { // DEV mode
    } else if (GMode == 3) // HOST mode
      new_arg_types[i] |= OMP_TGT_MAPTYPE_HOST;
    else if (GMode == 4) // HYB mode
      new_arg_types[i] |= OMP_TGT_MAPTYPE_HYB;
    // cluster priority
    CP += Rank;
  }
  if (GMode > 0 || argList.size() == 0)
    return std::make_pair(new_arg_types, new_arg_sizes);

  std::sort(argList.begin(), argList.end(), compareRank);
  // look up cluster
  if (!data_region) {
    Device.CurrentCluster = Device.lookupCluster(host_ptr);
    if (Device.CurrentCluster) {
      Device.IsNewCluster = false;
    } else {
      Device.CurrentCluster = new DataClusterTy(host_ptr);
      Device.DataClusters.push_front(*Device.CurrentCluster);
      Device.IsNewCluster = true;
    }
  } else
    Device.CurrentCluster = NULL;
  if (Device.CurrentCluster)
    Device.CurrentCluster->Priority = CP / argList.size();

  // fix data size and map type
  uint64_t CSize = 0;
  uint64_t RSize = 0;
  LookupResult *LRs = (LookupResult*)malloc(sizeof(LookupResult)*arg_num);
  for (auto I : argList) {
    int32_t idx = I.first;
    uint64_t DataSize = arg_sizes[idx];
    LookupResult lr = Device.lookupMapping(args_base[idx], DataSize);
    if ((lr.Flags.IsContained || lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter) ||
        (lr.Flags.InvalidContained || lr.Flags.InvalidExtendsB || lr.Flags.InvalidExtendsA))
      DataSize = lr.Entry->HstPtrEnd - lr.Entry->HstPtrBegin;
    new_arg_sizes[idx] = DataSize;
    if (lr.Entry != Device.HostDataToTargetMap.end() &&
        (!lr.Entry->Decided || !lr.Entry->IsValid)) {
      // restore recorded maptype
      new_arg_types[idx] &= ~0x3ff;
      new_arg_types[idx] |= lr.Entry->MapType & 0x3ff;
    }
    // find out the required space for this cluster
    if (lr.Entry == Device.HostDataToTargetMap.end() || !lr.Entry->IsValid ||
        ((lr.Entry->MapType & OMP_TGT_MAPTYPE_HOST) == OMP_TGT_MAPTYPE_HOST))
      RSize += DataSize;
    else
      lr.Entry->Irreplaceable = true;
    LRs[idx] = lr;
    CSize += DataSize;
  }

  // cluster mapping
  if (RSize > 0 && Device.CurrentCluster) {
    if (Device.IsNewCluster)
      Device.CurrentCluster->Size = CSize;
    else
      assert(Device.CurrentCluster->Size == CSize && "The size of cluster should be consistent.");
    replaceDataCluster(Device, RSize, total_dev_size - Device.deviceSize - Device.umSize, argList, LRs, new_arg_types, new_arg_sizes, args_base, ltc);
  } else if (RSize > 0) {
    for (auto I : argList) {
      int32_t idx = I.first;
      LookupResult lr = LRs[idx];
      if (lr.Entry != Device.HostDataToTargetMap.end() && lr.Entry->Irreplaceable)
        continue;
      LLD_DP("  Arg %d (" DPxMOD ") mapping is not decided\n", idx, DPxPTR(args_base[idx]));
      new_arg_types[idx] |= OMP_TGT_MAPTYPE_UVM;
      new_arg_types[idx] |= OMP_TGT_MAPTYPE_HOST;
    }
  }
  free(LRs);
  return std::make_pair(new_arg_types, new_arg_sizes);

  // arguments mapping
  for (auto I : argList) {
    int32_t idx = I.first;
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
    if (lr.Entry != Device.HostDataToTargetMap.end() && lr.Entry->IsValid &&
        ((lr.Entry->MapType & OMP_TGT_MAPTYPE_HOST) != OMP_TGT_MAPTYPE_HOST)) {
      continue;
    } else if (used_dev_size + Device.deviceSize + Device.umSize + DataSize <= total_dev_size) {
      placeDataObj(Device, &(*lr.Entry), idx, new_arg_types[idx], DataSize, args_base[idx], ltc, data_region);
      used_dev_size += DataSize;
    } else
      if (replaceDataObj(Device, &(*lr.Entry), idx, new_arg_types[idx], DataSize, total_dev_size - used_dev_size - Device.deviceSize - Device.umSize, args_base[idx], ltc, data_region))
        used_dev_size += DataSize;
    LLD_DP("  Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
        ", Type=0x%" PRIx64 "\n", idx, DPxPTR(args_base[idx]), DPxPTR(args[idx]),
        new_arg_sizes[idx], new_arg_types[idx]);
  }
  return std::make_pair(new_arg_types, new_arg_sizes);
}

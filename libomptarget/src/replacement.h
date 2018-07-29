// lld: utilities for replacement

#define REUSE_DIST_CENTRIC
#define NO_ON_DEMAND

/// Data object cluster
enum data_cluster_type {
  CLUSTER_MAPTYPE_DEV = 0,
  CLUSTER_MAPTYPE_MIX,
  CLUSTER_MAPTYPE_HST
};

struct DataClusterTy {
  void *BasePtr;
  std::list<HostDataToTargetTy*> Members;
  data_cluster_type Type = CLUSTER_MAPTYPE_MIX;
  uint64_t Size = 0;
  double Priority = 0.0;

  DataClusterTy(void *Base) : BasePtr(Base) {}
};

/// Data attributes for each data reference used in an OpenMP target region.
enum mem_map_type {
  MEM_MAPTYPE_DEV = 0,
  MEM_MAPTYPE_SDEV,
  MEM_MAPTYPE_UVM,
  MEM_MAPTYPE_PART,
  MEM_MAPTYPE_HOST,
  MEM_MAPTYPE_UNDECIDE
};

int64_t PartDevSize;

// get map type
inline mem_map_type getMemMapType(int64_t MapType) {
  bool IsUVM = MapType & OMP_TGT_MAPTYPE_UVM;
  bool IsHost = MapType & OMP_TGT_MAPTYPE_HOST;
  bool SoftDev = MapType & OMP_TGT_MAPTYPE_SDEV;
  bool Partial = MapType & OMP_TGT_MAPTYPE_PART;
  if (IsUVM & IsHost)
    return MEM_MAPTYPE_UNDECIDE;
  else if (IsUVM)
    return MEM_MAPTYPE_UVM;
  else if (IsHost)
    return MEM_MAPTYPE_HOST;
  else if (SoftDev)
    return MEM_MAPTYPE_SDEV;
  else if (Partial)
    return MEM_MAPTYPE_PART;
  else
    return MEM_MAPTYPE_DEV;
}

inline void setMemMapType(int64_t &MapType, mem_map_type Type) {
  MapType &= ~(OMP_TGT_MAPTYPE_UVM | OMP_TGT_MAPTYPE_HOST | OMP_TGT_MAPTYPE_SDEV | OMP_TGT_MAPTYPE_PART);
  if (Type == MEM_MAPTYPE_UNDECIDE)
    MapType |= (OMP_TGT_MAPTYPE_UVM | OMP_TGT_MAPTYPE_HOST);
  else if (Type == MEM_MAPTYPE_UVM)
    MapType |= OMP_TGT_MAPTYPE_UVM;
  else if (Type == MEM_MAPTYPE_HOST)
    MapType |= OMP_TGT_MAPTYPE_HOST;
  else if (Type == MEM_MAPTYPE_SDEV)
    MapType |= OMP_TGT_MAPTYPE_SDEV;
  else if (Type == MEM_MAPTYPE_PART)
    MapType |= OMP_TGT_MAPTYPE_PART;
  else
    assert(Type == MEM_MAPTYPE_DEV);
}

inline unsigned getLocalReuse(int64_t MapType) {
  return (MapType & OMP_TGT_MAPTYPE_LOCAL_REUSE) >> 20;
}

inline int64_t getGlobalReuse(int64_t MapType) {
  return (MapType & OMP_TGT_MAPTYPE_RANK) >> 12;
}

inline uint64_t getReuseDist(int64_t MapType) {
  return (MapType & OMP_TGT_MAPTYPE_DIST) >> 40;
}

bool isInDevCluster(HostDataToTargetTy *E) {
  for (auto *C : E->Clusters) {
    if (C->Type == CLUSTER_MAPTYPE_DEV)
      return true;
  }
  return false;
}

bool isInCluster(HostDataToTargetTy *E, DataClusterTy *CC) {
  for (auto *C : E->Clusters) {
    if (C == CC)
      return true;
  }
  return false;
}

// dump all target data
void dumpTargetData(HostDataToTargetListTy *DataList) {
  LLD_DP("Target data:\n");
  int i = 0;
  for (auto &HT : *DataList) {
    LLD_DP("Entry %2d: Base=" DPxMOD ", Valid=%d, Deleted=%d, Reuse=%ld, ReuseDist=%lu, Time=%lu, Size=%" PRId64
        ", DevSize=%" PRId64 " Type=0x%" PRIx64 "\n", i, DPxPTR(HT.HstPtrBegin), HT.IsValid, HT.IsDeleted, HT.Reuse, HT.ReuseDist, HT.TimeStamp,
        HT.HstPtrEnd - HT.HstPtrBegin, HT.DevSize, HT.MapType);
    i++;
  }
  LLD_DP("\n");
}

// replace a data object
int64_t placeDataObj(DeviceTy &Device, HostDataToTargetTy *Entry, int32_t idx, int64_t &MapType, int64_t Size, void *Base, uint64_t LTC, bool data_region) {
  if (data_region) {
    LLD_DP("  Arg %d (" DPxMOD ") mapping is not decided\n", idx, DPxPTR(Base));
    MapType |= OMP_TGT_MAPTYPE_UVM;
    MapType |= OMP_TGT_MAPTYPE_HOST;
  } else {
    unsigned LocalReuse = getLocalReuse(MapType);
    double LocalReuseFloat = (double)LocalReuse / 8.0;
    double Density = LocalReuseFloat * LTC / Size;
    if (Density < 0.5) {
      LLD_DP("  Arg %d (" DPxMOD ") is intended for UM (%f)\n", idx, DPxPTR(Base), Density);
#ifdef NO_ON_DEMAND
      MapType |= OMP_TGT_MAPTYPE_SDEV;
#else
      MapType |= OMP_TGT_MAPTYPE_UVM;
#endif
    } else {
      LLD_DP("  Arg %d (" DPxMOD ") is intended for device (%f)\n", idx, DPxPTR(Base), Density);
      MapType |= OMP_TGT_MAPTYPE_SDEV;
    }
  }
  if (Entry)
    Entry->ChangeMap = true;
  if (Entry && getMemMapType(Entry->MapType) == MEM_MAPTYPE_PART)
    return Size - Entry->DevSize;
  else
    return Size;
}

// release space
int64_t releaseDataObj(DeviceTy &Device, HostDataToTargetTy *E) {
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
  } else if (PreMap == MEM_MAPTYPE_PART) {
    Size = E->DevSize;
    LLD_DP("  Replace " DPxMOD " from part, size=%ld\n", DPxPTR(E->HstPtrBegin), Size);
    Device.RTL->data_opt(Device.RTLDeviceID, Size, (void *)E->HstPtrBegin, 0); // pin to host
    Device.RTL->data_opt(Device.RTLDeviceID, Size, (void *)E->HstPtrBegin, 5); // prefetch to host
    Device.deviceSize -= Size;
    setMemMapType(E->MapType, MEM_MAPTYPE_HOST);
  } else
    assert(0);
  return Size;
}

int64_t replaceDataObjPart(DeviceTy &Device, HostDataToTargetTy *Entry, int32_t idx, int64_t &MapType, int64_t Size, int64_t AvailSize, void *Base, uint64_t LTC, bool data_region) {
#ifdef REUSE_DIST_CENTRIC
  uint64_t ReuseDist = getReuseDist(MapType);
#else
  int64_t Reuse = getGlobalReuse(MapType);
#endif
  std::vector<HostDataToTargetTy*> ReplaceList;
  for (auto &HT : Device.HostDataToTargetMap) {
    // find objects with poorer locality
    mem_map_type PreMap = getMemMapType(HT.MapType);
#ifdef REUSE_DIST_CENTRIC
    if (HT.ReuseDist + HT.TimeStamp > GlobalTimeStamp + ReuseDist && PreMap == MEM_MAPTYPE_PART && Entry != &HT) {
#else
    if (HT.Reuse > Reuse && PreMap == MEM_MAPTYPE_PART) {
#endif
      int64_t HTSize = HT.DevSize;
      if (!HT.IsDeleted && HTSize >= 4096) { // Do not replace small objects
        AvailSize += HTSize;
        ReplaceList.push_back(&HT);
        // FIXME: did not consider combined locality
      }
    }
  }

  assert(AvailSize < Size);
  LLD_DP("  Not enough space for replacement (%ld < %ld, %lu obj)\n", AvailSize, Size, ReplaceList.size());
  if (AvailSize >= 4096) {
    // release space
    for (auto *E : ReplaceList)
      releaseDataObj(Device, E);
    // partial map
    mem_map_type PreMap = MEM_MAPTYPE_UNDECIDE;
    if (Entry)
      PreMap = getMemMapType(Entry->MapType);
    PartDevSize = AvailSize;
    if (PreMap == MEM_MAPTYPE_PART) {
      int64_t NewDevSize = AvailSize - Entry->DevSize;
      if (NewDevSize >= 4096) {
        LLD_DP("  Arg %d (" DPxMOD ") is remapped to part\n", idx, DPxPTR(Base));
        setMemMapType(MapType, MEM_MAPTYPE_PART);
        Entry->ChangeMap = true;
        return NewDevSize;
      } else
        return 0;
    } else {
      LLD_DP("  Arg %d (" DPxMOD ") is mapped to part\n", idx, DPxPTR(Base));
      setMemMapType(MapType, MEM_MAPTYPE_PART);
      if (Entry)
        Entry->ChangeMap = true;
      return AvailSize;
    }
  } else {
    mem_map_type PreMap = MEM_MAPTYPE_UNDECIDE;
    if (Entry)
      PreMap = getMemMapType(Entry->MapType);
    if (PreMap == MEM_MAPTYPE_UNDECIDE) {
      LLD_DP("  Arg %d (" DPxMOD ") is mapped to host\n", idx, DPxPTR(Base));
      setMemMapType(MapType, MEM_MAPTYPE_HOST);
    }
  }
  return 0;
}

// lld: compare reuse
bool compareCandidates(HostDataToTargetTy *A, HostDataToTargetTy *B) {
#ifdef REUSE_DIST_CENTRIC
  uint64_t AR = A->ReuseDist + A->TimeStamp;
  uint64_t BR = B->ReuseDist + B->TimeStamp;
  return (AR == BR) ? (A->Reuse > B->Reuse) : (AR > BR);
#else
  return (A->Reuse > B->Reuse);
#endif
}

// replace a data object
int64_t replaceDataObj(DeviceTy &Device, HostDataToTargetTy *Entry, int32_t idx, int64_t &MapType, int64_t Size, int64_t AvailSize, void *Base, uint64_t LTC, bool data_region) {
  int64_t OriAvailSize = AvailSize;
#ifdef REUSE_DIST_CENTRIC
  uint64_t ReuseDist = getReuseDist(MapType);
#else
  int64_t Reuse = getGlobalReuse(MapType);
#endif
  std::vector<HostDataToTargetTy*> ReplaceList;
  for (auto &HT : Device.HostDataToTargetMap) {
    // if it is not replaceable
    if (HT.Irreplaceable)
      continue;
    // if it belongs to a pinned cluster
    if (isInDevCluster(&HT))
      continue;
    // find objects with poorer locality
    mem_map_type PreMap = getMemMapType(HT.MapType);
#ifdef REUSE_DIST_CENTRIC
    if ((HT.ReuseDist + HT.TimeStamp > GlobalTimeStamp + ReuseDist && PreMap < MEM_MAPTYPE_PART) ||
#else
    if ((HT.Reuse > Reuse && PreMap < MEM_MAPTYPE_PART) ||
#endif
        (PreMap == MEM_MAPTYPE_PART && Entry != &HT)) { // implicitly assume at most 1 is mapped to part
      int64_t HTSize;
      if (PreMap == MEM_MAPTYPE_PART)
        HTSize = HT.DevSize;
      else
        HTSize = HT.HstPtrEnd - HT.HstPtrBegin;
      if (!HT.IsDeleted && HTSize >= 4096) { // Do not replace small objects
        AvailSize += HTSize;
        ReplaceList.push_back(&HT);
        // FIXME: did not consider combined locality
        // FIXME: did not consider whether enough space is released
      }
    }
  }

  // sort replace list based on reuse
  std::sort(ReplaceList.begin(), ReplaceList.end(), compareCandidates);

  if (AvailSize < Size) {
    LLD_DP("  Not enough space for replacement (%ld < %ld, %lu obj)\n", AvailSize, Size, ReplaceList.size());
    mem_map_type PreMap = MEM_MAPTYPE_UNDECIDE;
    if (Entry)
      PreMap = getMemMapType(Entry->MapType);
    if (PreMap == MEM_MAPTYPE_UNDECIDE) {
      LLD_DP("  Arg %d (" DPxMOD ") is intended for host\n", idx, DPxPTR(Base));
      setMemMapType(MapType, MEM_MAPTYPE_HOST);
    }
    return 0;
  }

  // release space
  AvailSize = OriAvailSize;
  for (auto *E : ReplaceList) {
    AvailSize += releaseDataObj(Device, E);
    if (AvailSize >= Size)
      break;
  }
  // place data
  return placeDataObj(Device, Entry, idx, MapType, Size, Base, LTC, data_region);
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
  //LLD_DP("  Cluster " DPxMOD ", size=%ld, avail=%ld\n", DPxPTR(Device.CurrentCluster->BasePtr), Size, AvailSize);
  int64_t OriAvailSize = AvailSize;
  if (AvailSize >= Size) {
    LLD_DP("  Cluster " DPxMOD " uses device mapping\n", DPxPTR(Device.CurrentCluster->BasePtr));
    Device.CurrentCluster->Type = CLUSTER_MAPTYPE_DEV;
    for (auto I : argList) {
      int32_t idx = I.first;
      LookupResult lr = LRs[idx];
      if (lr.Entry != Device.HostDataToTargetMap.end() && lr.Entry->Irreplaceable) {
        lr.Entry->Irreplaceable = false;
        continue;
      }
      HostDataToTargetTy *HT = (lr.Entry != Device.HostDataToTargetMap.end() ? &(*lr.Entry) : NULL);
      placeDataObj(Device, HT, idx, argTypes[idx], argSizes[idx], argBases[idx], LTC, false);
    }
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
    // if it belongs to the current cluster
    if (isInCluster(&HT, Device.CurrentCluster))
      continue;
    // find objects with poorer locality
    mem_map_type PreMap = getMemMapType(HT.MapType);
    if (PreMap <= MEM_MAPTYPE_PART) {
      int64_t HTSize;
      if (PreMap == MEM_MAPTYPE_PART)
        HTSize = HT.DevSize;
      else
        HTSize = HT.HstPtrEnd - HT.HstPtrBegin;
      if (!HT.IsDeleted && HTSize >= 4096) { // Do not replace small objects
        AvailSize += HTSize;
        ReplaceList.push_back(&HT);
      }
    }
  }

  if (AvailSize < Size) {
    LLD_DP("  Cluster " DPxMOD " uses mixed mapping\n", DPxPTR(Device.CurrentCluster->BasePtr));
    Device.CurrentCluster->Type = CLUSTER_MAPTYPE_MIX;
    int64_t used_dev_size = 0;
    // argument index for partial mapping
    int32_t partial_idx = -1;
    HostDataToTargetTy *partial_HT;
    for (auto I : argList) {
      int32_t idx = I.first;
      LookupResult lr = LRs[idx];
      HostDataToTargetTy *HT = (lr.Entry != Device.HostDataToTargetMap.end() ? &(*lr.Entry) : NULL);
      if (HT && HT->Irreplaceable) {
        HT->Irreplaceable = false;
        continue;
      }
      AvailSize = total_dev_size - used_dev_size - Device.deviceSize - Device.umSize;
      assert(AvailSize > -1024); // reserve space for non UM variables
      if (HT && getMemMapType(HT->MapType) == MEM_MAPTYPE_PART)
        AvailSize += HT->DevSize;
      if (argSizes[idx] <= AvailSize)
        used_dev_size += placeDataObj(Device, HT, idx, argTypes[idx], argSizes[idx], argBases[idx], LTC, false);
      else {
        int64_t allocateSize = replaceDataObj(Device, HT, idx, argTypes[idx], argSizes[idx], AvailSize, argBases[idx], LTC, false);
        used_dev_size += allocateSize;
        if (PartialMap && allocateSize == 0 && partial_idx == -1) {
          partial_idx = idx;
          partial_HT = HT;
        }
      }
    }
    if (PartialMap && partial_idx >= 0) {
      AvailSize = total_dev_size - used_dev_size - Device.deviceSize - Device.umSize;
      if (partial_HT && getMemMapType(partial_HT->MapType) == MEM_MAPTYPE_PART)
        AvailSize += partial_HT->DevSize;
      replaceDataObjPart(Device, partial_HT, partial_idx, argTypes[partial_idx], argSizes[partial_idx], AvailSize, argBases[partial_idx], LTC, false);
    }
  } else {
    LLD_DP("  Cluster " DPxMOD " uses device mapping\n", DPxPTR(Device.CurrentCluster->BasePtr));
    Device.CurrentCluster->Type = CLUSTER_MAPTYPE_DEV;
    AvailSize = OriAvailSize;
    // sort replace list based on reuse
    std::sort(ReplaceList.begin(), ReplaceList.end(), compareCandidates);
    for (auto *E : ReplaceList) {
      AvailSize += releaseDataObj(Device, E);
      if (AvailSize >= Size)
        break;
    }
    for (auto I : argList) {
      int32_t idx = I.first;
      LookupResult lr = LRs[idx];
      HostDataToTargetTy *HT = (lr.Entry != Device.HostDataToTargetMap.end() ? &(*lr.Entry) : NULL);
      if (HT && HT->Irreplaceable) {
        HT->Irreplaceable = false;
        continue;
      }
      placeDataObj(Device, HT, idx, argTypes[idx], argSizes[idx], argBases[idx], LTC, false);
    }
  }
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
  mem_map_type CurMap = getMemMapType(MapType);

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
      } else if (CurMap == MEM_MAPTYPE_PART) {
        LLD_DP("  Map " DPxMOD " to part, size=%ld (%ld)\n", DPxPTR(HstPtrBegin), Size, PartDevSize);
        HT.TgtPtrBegin = (uintptr_t)HstPtrBegin;
        RTL->data_opt(RTLDeviceID, PartDevSize, HstPtrBegin, 4); // pin to device
        RTL->data_opt(RTLDeviceID, PartDevSize, HstPtrBegin, 1); // prefetch
        deviceSize += PartDevSize;
        RTL->data_opt(RTLDeviceID, Size-PartDevSize, (void*)((uintptr_t)HstPtrBegin+PartDevSize), 0); // pin to host
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
        } else if (PreMap == MEM_MAPTYPE_PART) {
          LLD_DP("  Remap " DPxMOD " from part to UM, size=%ld\n", DPxPTR(HstPtrBegin), Size);
          RTL->data_opt(RTLDeviceID, Size, HstPtrBegin, 6); // unpin
          deviceSize -= HT.DevSize;
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
        } else if (PreMap == MEM_MAPTYPE_PART) {
          LLD_DP("  Remap " DPxMOD " from part to soft device, size=%ld\n", DPxPTR(HstPtrBegin), Size);
          RTL->data_opt(RTLDeviceID, Size-HT.DevSize, (void*)((uintptr_t)HstPtrBegin+HT.DevSize), 4); // pin to device
          RTL->data_opt(RTLDeviceID, Size-HT.DevSize, (void*)((uintptr_t)HstPtrBegin+HT.DevSize), 1); // prefetch
          deviceSize += Size-HT.DevSize;
        }
      } else if (CurMap == MEM_MAPTYPE_PART) {
        if (PreMap == MEM_MAPTYPE_DEV) {
          assert(HT.TgtPtrBegin != HT.HstPtrBegin);
          LLD_DP("  Remap " DPxMOD " from device (" DPxMOD ") to part, size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size);
          RTL->data_delete(RTLDeviceID, (void *)HT.TgtPtrBegin);
          HT.TgtPtrBegin = (uintptr_t)HstPtrBegin;
          RTL->data_opt(RTLDeviceID, PartDevSize, HstPtrBegin, 4); // pin to device
          RTL->data_opt(RTLDeviceID, PartDevSize, HstPtrBegin, 1); // prefetch
          RTL->data_opt(RTLDeviceID, Size-PartDevSize, (void*)((uintptr_t)HstPtrBegin+PartDevSize), 0); // pin to host
          RTL->data_opt(RTLDeviceID, Size-PartDevSize, (void*)((uintptr_t)HstPtrBegin+PartDevSize), 5); // prefetch to host
          deviceSize -= Size-PartDevSize;
        } else if (PreMap == MEM_MAPTYPE_UVM) {
          LLD_DP("  Remap " DPxMOD " from UM to part, size=%ld\n", DPxPTR(HstPtrBegin), Size);
          RTL->data_opt(RTLDeviceID, PartDevSize, HstPtrBegin, 4); // pin to device
          RTL->data_opt(RTLDeviceID, PartDevSize, HstPtrBegin, 1); // prefetch
          RTL->data_opt(RTLDeviceID, Size-PartDevSize, (void*)((uintptr_t)HstPtrBegin+PartDevSize), 0); // pin to host
          RTL->data_opt(RTLDeviceID, Size-PartDevSize, (void*)((uintptr_t)HstPtrBegin+PartDevSize), 5); // prefetch to host
          umSize -= Size;
          deviceSize += PartDevSize;
        } else if (PreMap == MEM_MAPTYPE_HOST) {
          LLD_DP("  Remap " DPxMOD " from host to part, size=%ld\n", DPxPTR(HstPtrBegin), Size);
          RTL->data_opt(RTLDeviceID, PartDevSize, HstPtrBegin, 4); // pin to device
          RTL->data_opt(RTLDeviceID, PartDevSize, HstPtrBegin, 1); // prefetch
          deviceSize += PartDevSize;
        } else if (PreMap == MEM_MAPTYPE_SDEV) {
          LLD_DP("  Remap " DPxMOD " from soft device to part, size=%ld\n", DPxPTR(HstPtrBegin), Size);
          RTL->data_opt(RTLDeviceID, Size-PartDevSize, (void*)((uintptr_t)HstPtrBegin+PartDevSize), 0); // pin to host
          RTL->data_opt(RTLDeviceID, Size-PartDevSize, (void*)((uintptr_t)HstPtrBegin+PartDevSize), 5); // prefetch to host
          deviceSize -= Size-PartDevSize;
        } else if (PreMap == MEM_MAPTYPE_PART) {
          LLD_DP("  Remap " DPxMOD " from part to part, size=%ld\n", DPxPTR(HstPtrBegin), Size);
          // FIXME: can be optimized
          RTL->data_opt(RTLDeviceID, PartDevSize, HstPtrBegin, 4); // pin to device
          RTL->data_opt(RTLDeviceID, PartDevSize, HstPtrBegin, 1); // prefetch
          RTL->data_opt(RTLDeviceID, Size-PartDevSize, (void*)((uintptr_t)HstPtrBegin+PartDevSize), 0); // pin to host
          RTL->data_opt(RTLDeviceID, Size-PartDevSize, (void*)((uintptr_t)HstPtrBegin+PartDevSize), 5); // prefetch to host
          deviceSize += PartDevSize - HT.DevSize;
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
        } else if (PreMap == MEM_MAPTYPE_PART) {
          LLD_DP("  Remap " DPxMOD " from part to host, size=%ld\n", DPxPTR(HstPtrBegin), Size);
          RTL->data_opt(RTLDeviceID, HT.DevSize, HstPtrBegin, 0); // pin to host
          RTL->data_opt(RTLDeviceID, HT.DevSize, HstPtrBegin, 5); // prefetch to host
          deviceSize -= HT.DevSize;
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
        } else if (PreMap == MEM_MAPTYPE_PART) {
          RTL->data_opt(RTLDeviceID, HT.DevSize, HstPtrBegin, 0); // pin to host
          RTL->data_opt(RTLDeviceID, HT.DevSize, HstPtrBegin, 5); // prefetch to host
          HT.TgtPtrBegin = (uintptr_t)RTL->data_alloc(RTLDeviceID, Size, HstPtrBegin);
          int rt = RTL->data_submit(RTLDeviceID, (void*)HT.TgtPtrBegin, HstPtrBegin, Size);
          if (rt != OFFLOAD_SUCCESS)
            LLD_DP("Copying data to device failed.\n");
          LLD_DP("  Remap " DPxMOD " from part to device (" DPxMOD "), size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size);
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
    } else if (CurMap == MEM_MAPTYPE_PART) {
      if (HT.TgtPtrBegin != HT.HstPtrBegin) {
        deviceSize -= Size;
        RTL->data_delete(RTLDeviceID, (void *)HT.TgtPtrBegin);
        LLD_DP("  Unmap " DPxMOD " from device (" DPxMOD "), size=%ld\n", DPxPTR(HstPtrBegin), DPxPTR(HT.TgtPtrBegin), Size);
      }
      LLD_DP("  Remap " DPxMOD " to part, size=%ld (%ld)\n", DPxPTR(HstPtrBegin), Size, PartDevSize);
      tp = (uintptr_t)HstPtrBegin;
      RTL->data_opt(RTLDeviceID, PartDevSize, HstPtrBegin, 4); // pin to device
      RTL->data_opt(RTLDeviceID, PartDevSize, HstPtrBegin, 1); // prefetch
      deviceSize += PartDevSize;
      RTL->data_opt(RTLDeviceID, Size-PartDevSize, (void*)(tp+PartDevSize), 0); // pin to host
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
    } else if (CurMap == MEM_MAPTYPE_PART) {
      LLD_DP("  Map " DPxMOD " to part, size=%ld (%ld)\n", DPxPTR(HstPtrBegin), Size, PartDevSize);
      tp = (uintptr_t)HstPtrBegin;
      RTL->data_opt(RTLDeviceID, PartDevSize, HstPtrBegin, 4); // pin to device
      RTL->data_opt(RTLDeviceID, PartDevSize, HstPtrBegin, 1); // prefetch
      deviceSize += PartDevSize;
      RTL->data_opt(RTLDeviceID, Size-PartDevSize, (void*)(tp+PartDevSize), 0); // pin to host
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
        deviceSize += DevSize;
        RTL->data_opt(RTLDeviceID, Size-DevSize, (void*)(tp+DevSize), 0); // pin to host
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
    DataEntry.Reuse = getGlobalReuse(MapType);
    HostDataToTargetMap.push_front(DataEntry);
    DMEP = &(*HostDataToTargetMap.begin());
    // lld: insert to cluster
    if (CurrentCluster && IsNewCluster) {
      CurrentCluster->Members.push_front(DMEP);
      DMEP->Clusters.push_front(CurrentCluster);
    }
    rc = (void *)tp;
  }

  DataMapMtx.unlock();
#ifdef LLD_VERBOSE
  if (DMEP->ReuseDist > 0 && CurrentCluster) {
    if (DMEP->ReuseDist + DMEP->TimeStamp == GlobalTimeStamp) {
      LLD_DP("  Reuse distance hit\n");
    } else {
      LLD_DP("  Reuse distance miss: %lu + %lu != %lu\n", DMEP->TimeStamp, DMEP->ReuseDist, GlobalTimeStamp);
    }
  }
#endif
  // lld: update replacement info
  DMEP->TimeStamp = GlobalTimeStamp;
  DMEP->ReuseDist = getReuseDist(MapType);
  if (CurMap == MEM_MAPTYPE_PART)
    DMEP->DevSize = PartDevSize;
  return rc;
}

// lld: compare rank
bool rankArgs(std::pair<int32_t, int64_t> A, std::pair<int32_t, int64_t> B) {
  int64_t AG = getGlobalReuse(A.second);
  int64_t BG = getGlobalReuse(B.second);
#ifdef REUSE_DIST_CENTRIC
  int64_t ARD = getReuseDist(A.second);
  int64_t BRD = getReuseDist(B.second);
  return (ARD == BRD) ? (AG < BG) : (ARD < BRD);
#else
  return (AG < BG);
#endif
}

// lld: decide mapping based on rank
std::pair<int64_t*, int64_t*> target_uvm_data_mapping_opt(DeviceTy &Device, void **args_base, void **args, int32_t arg_num, int64_t *arg_sizes, int64_t *arg_types, void *host_ptr) {
#ifdef LLD_VERBOSE
  dumpTargetData(&Device.HostDataToTargetMap);
#endif
  for (int i=0; i<arg_num; ++i) {
    LLD_DP("Entry %2d: Base=" DPxMOD ", Begin=" DPxMOD ", Size=%" PRId64
        ", Type=0x%" PRIx64 "\n", i, DPxPTR(args_base[i]), DPxPTR(args[i]),
        arg_sizes[i], arg_types[i]);
  }
  GlobalTimeStamp++;
  int64_t used_dev_size = 0;
  uint64_t ltc = Device.loopTripCnt;
  bool data_region = (host_ptr == NULL ? true : false);
  if (data_region)
    LLD_DP("DATA\t\t\t\t(#iter: %lu    device: %ld    UM: %ld) at %lu\n", ltc, Device.deviceSize, Device.umSize, GlobalTimeStamp)
  else
    LLD_DP("COMPUTE (" DPxMOD ")\t(#iter: %lu    device: %ld    UM: %ld) at %lu\n", DPxPTR(host_ptr), ltc, Device.deviceSize, Device.umSize, GlobalTimeStamp)

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
    argList.push_back(std::make_pair(i, arg_types[i]));
    if (GMode == 1) // UM mode
      new_arg_types[i] |= OMP_TGT_MAPTYPE_UVM;
    else if (GMode == 2) { // DEV mode
    } else if (GMode == 3) // HOST mode
      new_arg_types[i] |= OMP_TGT_MAPTYPE_HOST;
    else if (GMode == 4) // HYB mode
      new_arg_types[i] |= OMP_TGT_MAPTYPE_HYB;
    else if (GMode == 5) // SDEV mode
      new_arg_types[i] |= OMP_TGT_MAPTYPE_SDEV;
    // cluster priority
    CP += getGlobalReuse(arg_types[i]);
  }
  if (GMode > 0 || argList.size() == 0)
    return std::make_pair(new_arg_types, new_arg_sizes);

  std::sort(argList.begin(), argList.end(), rankArgs);
  // look up cluster
  if (!data_region) {
    Device.CurrentCluster = Device.lookupCluster(host_ptr);
    if (Device.CurrentCluster) {
      Device.IsNewCluster = false;
    } else {
      Device.DataClusters.push_front(DataClusterTy(host_ptr));
      Device.CurrentCluster = &*(Device.DataClusters.begin());
      Device.IsNewCluster = true;
    }
  } else
    Device.CurrentCluster = NULL;
  if (Device.CurrentCluster)
    Device.CurrentCluster->Priority = CP / argList.size();

  if (GMode == 0) {
    // fix data size and map type
    uint64_t CSize = 0;
    uint64_t RSize = 0;
    LookupResult *LRs = (LookupResult*)malloc(sizeof(LookupResult)*arg_num);
    for (auto I : argList) {
      int32_t idx = I.first;
      int64_t DataSize = arg_sizes[idx];
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
      // lld: insert to cluster
      if (lr.Entry != Device.HostDataToTargetMap.end() &&
          Device.CurrentCluster) {
        lr.Entry->Clusters.push_front(Device.CurrentCluster);
        Device.CurrentCluster->Members.push_front(&*(lr.Entry));
      }
      // find out the required space for this cluster
      if (lr.Entry == Device.HostDataToTargetMap.end() || !lr.Entry->IsValid ||
          getMemMapType(lr.Entry->MapType) >= MEM_MAPTYPE_HOST)
        RSize += DataSize;
      else if (lr.Entry != Device.HostDataToTargetMap.end() && lr.Entry->IsValid &&
               getMemMapType(lr.Entry->MapType) == MEM_MAPTYPE_PART)
        RSize += DataSize - lr.Entry->DevSize;
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
        if (lr.Entry != Device.HostDataToTargetMap.end() && lr.Entry->Irreplaceable) {
          lr.Entry->Irreplaceable = false;
          continue;
        }
        LLD_DP("  Arg %d (" DPxMOD ") mapping is not decided\n", idx, DPxPTR(args_base[idx]));
        new_arg_types[idx] |= OMP_TGT_MAPTYPE_UVM;
        new_arg_types[idx] |= OMP_TGT_MAPTYPE_HOST;
      }
    }
    free(LRs);
  } else {
    // argument index for partial mapping
    int32_t partial_idx = -1;
    HostDataToTargetTy *partial_HT;
    // arguments mapping
    for (auto I : argList) {
      int32_t idx = I.first;
      int64_t DataSize = arg_sizes[idx];
      LookupResult lr = Device.lookupMapping(args_base[idx], DataSize);
      //if (lr.Flags.IsContained || lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter)
      if ((lr.Flags.IsContained || lr.Flags.ExtendsBefore || lr.Flags.ExtendsAfter) ||
          (lr.Flags.InvalidContained || lr.Flags.InvalidExtendsB || lr.Flags.InvalidExtendsA))
        DataSize = lr.Entry->HstPtrEnd - lr.Entry->HstPtrBegin;
      assert(DataSize > 0);
      if (lr.Entry != Device.HostDataToTargetMap.end() &&
          (!lr.Entry->Decided || !lr.Entry->IsValid)) {
        // restore recorded maptype
        new_arg_types[idx] &= ~0x3ff;
        new_arg_types[idx] |= lr.Entry->MapType & 0x3ff;
      }
      // lld: insert to cluster
      if (lr.Entry != Device.HostDataToTargetMap.end() &&
          Device.CurrentCluster) {
        lr.Entry->Clusters.push_front(Device.CurrentCluster);
        Device.CurrentCluster->Members.push_front(&*(lr.Entry));
      }
      // restore size
      new_arg_sizes[idx] = DataSize;
      HostDataToTargetTy *HT = (lr.Entry != Device.HostDataToTargetMap.end() ? &(*lr.Entry) : NULL);
      if (HT && getMemMapType(HT->MapType) != MEM_MAPTYPE_PART)
        lr.Entry->Irreplaceable = true;
      if (HT && HT->IsValid && (getMemMapType(HT->MapType) <= MEM_MAPTYPE_UVM))
        continue;
      int64_t AvailSize = total_dev_size - used_dev_size - Device.deviceSize - Device.umSize;
      assert(AvailSize > -1024); // reserve space for non UM variables
      if (HT && getMemMapType(HT->MapType) == MEM_MAPTYPE_PART)
        AvailSize += HT->DevSize;
      if (DataSize <= AvailSize)
        used_dev_size += placeDataObj(Device, HT, idx, new_arg_types[idx], DataSize, args_base[idx], ltc, data_region);
      else {
        int64_t allocateSize = replaceDataObj(Device, HT, idx, new_arg_types[idx], DataSize, AvailSize, args_base[idx], ltc, data_region);
        used_dev_size += allocateSize;
        if (PartialMap && allocateSize == 0 && partial_idx == -1) {
          partial_idx = idx;
          partial_HT = HT;
        }
      }
    }
    if (PartialMap && partial_idx >= 0) {
      int64_t AvailSize = total_dev_size - used_dev_size - Device.deviceSize - Device.umSize;
      if (partial_HT && getMemMapType(partial_HT->MapType) == MEM_MAPTYPE_PART)
        AvailSize += partial_HT->DevSize;
      replaceDataObjPart(Device, partial_HT, partial_idx, new_arg_types[partial_idx], new_arg_sizes[partial_idx], AvailSize, args_base[partial_idx], ltc, data_region);
    }
    cleanReplaceMetadata(Device);
  }
  return std::make_pair(new_arg_types, new_arg_sizes);
}

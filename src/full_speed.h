// Keeps the process out of Windows power throttling (EcoQoS). Windows 11 may run a process it considers background
// work on the efficiency cores at low clocks for its whole life; the engine's host thread then answers every GPU
// round trip late. Speculative decoding (a read-back per draft and per verification) suffered most: server sessions
// of Qwen3.6 35B with MTP at 12-17 tok/s instead of 26 at random, while the same requests in other sessions were fast.
// ARC_ECOQOS=1 leaves the default (A/B tests).
#pragma once
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdlib>

inline void full_speed_process() {
  if (const char* e = getenv("ARC_ECOQOS"); e && e[0] == '1') return;
  PROCESS_POWER_THROTTLING_STATE s = {};
  s.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
  s.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
  s.StateMask = 0;  // (the controlled policies off: full speed, timer resolution honored)
  SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &s, sizeof s);
}

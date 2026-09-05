/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import {smoothPackedValues, yieldToEventLoop} from './smoothing_kernel';

const jobs = new Map<number, {cancelled: boolean}>();
self.addEventListener('message', async (event: MessageEvent) => {
  const {id, cancel, buffer, lengths, weight} = event.data;
  if (cancel) {
    const job = jobs.get(id);
    if (job) job.cancelled = true;
    return;
  }
  const job = {cancelled: false};
  jobs.set(id, job);
  try {
    await smoothPackedValues(
      new Float64Array(buffer),
      lengths,
      weight,
      () => job.cancelled,
      yieldToEventLoop
    );
    if (!job.cancelled)
      (self as unknown as Worker).postMessage({id, buffer}, [buffer]);
  } catch {
    if (!job.cancelled)
      (self as unknown as Worker).postMessage({id, error: true});
  } finally {
    jobs.delete(id);
  }
});

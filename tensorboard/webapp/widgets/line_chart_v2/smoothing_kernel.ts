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

let yieldChannel: MessageChannel | undefined;
const yielded: Array<() => void> = [];

/** Yield without the nested setTimeout minimum delay. */
export function yieldToEventLoop(): Promise<void> {
  if (!yieldChannel) {
    yieldChannel = new MessageChannel();
    yieldChannel.port1.onmessage = () => yielded.shift()?.();
  }
  return new Promise((resolve) => {
    yielded.push(resolve);
    yieldChannel!.port2.postMessage(null);
  });
}

/** In-place classic EMA over packed y columns, with cooperative cancellation. */
export async function smoothPackedValues(
  values: Float64Array,
  lengths: number[],
  weight: number,
  cancelled: () => boolean = () => false,
  yieldTask?: () => Promise<void>
): Promise<void> {
  let offset = 0;
  let work = 0;
  async function checkpoint() {
    if (yieldTask) await yieldTask();
    if (cancelled()) throw new Error('Smoothing cancelled');
  }
  for (const length of lengths) {
    if (cancelled()) throw new Error('Smoothing cancelled');
    const end = offset + length;
    const initial = values[offset];
    let constant = true;
    for (let i = offset; i < end; i++) {
      if (values[i] !== initial) {
        constant = false;
        break;
      }
      if (++work % 4096 === 0 && yieldTask) await checkpoint();
    }
    if (!constant) {
      let last = 0;
      let count = 0;
      for (let i = offset; i < end; i++) {
        const value = values[i];
        if (Number.isFinite(value)) {
          last = last * weight + (1 - weight) * value;
          count++;
          const debias = weight === 1 ? 1 : 1 - Math.pow(weight, count);
          values[i] = last / debias;
        }
        if (++work % 4096 === 0 && yieldTask) await checkpoint();
      }
    }
    offset = end;
  }
}

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

import {getWorker} from './lib/worker';
import {smoothPackedValues, yieldToEventLoop} from './smoothing_kernel';
import {SMOOTHING_WORKER_PATH} from './smoothing_resource';

/** One shared worker; injectable factory allows exercising failure/cancellation. */
export class SmoothingWorkerClient {
  private worker: Worker | undefined;
  private workerFailed = false;
  private nextId = 0;
  private readonly jobs = new Map<
    number,
    {resolve: (values: Float64Array) => void; reject: () => void}
  >();

  constructor(private readonly factory: typeof getWorker = getWorker) {}

  private getWorker(): Worker {
    if (this.workerFailed) throw new Error('Smoothing worker unavailable');
    if (!this.worker) {
      try {
        this.worker = this.factory(SMOOTHING_WORKER_PATH);
      } catch (error) {
        this.workerFailed = true;
        throw error;
      }
      this.worker.addEventListener('message', ({data}: MessageEvent) => {
        const job = this.jobs.get(data.id);
        if (!job) return;
        this.jobs.delete(data.id);
        if (data.error) job.reject();
        else job.resolve(new Float64Array(data.buffer));
      });
      const fail = () => {
        this.workerFailed = true;
        this.worker?.terminate();
        this.worker = undefined;
        for (const job of this.jobs.values()) job.reject();
        this.jobs.clear();
      };
      this.worker.addEventListener('error', fail);
      this.worker.addEventListener('messageerror', fail);
    }
    return this.worker;
  }

  /** Transfer large jobs; failed/unavailable workers use a yielding fallback. */
  async smooth(
    makeValues: () => Float64Array,
    lengths: number[],
    weight: number,
    signal?: AbortSignal
  ): Promise<Float64Array> {
    if (signal?.aborted) throw new Error('Smoothing cancelled');
    const values = makeValues();
    const large = values.length >= 8192;
    if (large && !this.workerFailed) {
      try {
        const target = this.getWorker();
        const id = ++this.nextId;
        return await new Promise<Float64Array>((resolve, reject) => {
          const cleanup = () => signal?.removeEventListener('abort', abort);
          const abort = () => {
            this.jobs.delete(id);
            target.postMessage({id, cancel: true});
            cleanup();
            reject(new Error('Smoothing cancelled'));
          };
          this.jobs.set(id, {
            resolve: (result) => {
              cleanup();
              resolve(result);
            },
            reject: () => {
              cleanup();
              reject(new Error('Smoothing worker failed'));
            },
          });
          signal?.addEventListener('abort', abort, {once: true});
          try {
            target.postMessage({id, buffer: values.buffer, lengths, weight}, [
              values.buffer,
            ]);
          } catch (error) {
            this.jobs.delete(id);
            cleanup();
            reject(error);
          }
        });
      } catch {
        if (signal?.aborted) throw new Error('Smoothing cancelled');
        // A transfer may have detached the input. Rebuild from immutable points.
        const fallback = makeValues();
        await smoothPackedValues(
          fallback,
          lengths,
          weight,
          () => !!signal?.aborted,
          yieldToEventLoop
        );
        return fallback;
      }
    }
    await smoothPackedValues(
      values,
      lengths,
      weight,
      () => !!signal?.aborted,
      large ? yieldToEventLoop : undefined
    );
    return values;
  }
}

const smoothingClient = new SmoothingWorkerClient();
export const smoothOffThread = smoothingClient.smooth.bind(smoothingClient);

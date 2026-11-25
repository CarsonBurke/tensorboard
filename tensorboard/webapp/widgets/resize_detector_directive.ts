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
import {
  Directive,
  ElementRef,
  EventEmitter,
  Input,
  OnDestroy,
  OnInit,
  Output,
} from '@angular/core';
import {Subject} from 'rxjs';
import {debounceTime, skip, takeUntil} from 'rxjs/operators';

/**
 * A directive that calls `onResize` when a host element is resized.
 *
 * By default, it does not emit `onResize` on the initial render.
 */
@Directive({
  standalone: false,
  selector: '[detectResize]',
})
export class ResizeDetectorDirective implements OnDestroy, OnInit {
  @Input() resizeEventDebouncePeriodInMs: number = 100;
  @Input() emitInitialResize: boolean = false;
  @Output() onResize = new EventEmitter<void>();

  private readonly ngUnsubscribe$ = new Subject<void>();
  private readonly onResize$ = new Subject<void>();

  constructor(ref: ElementRef) {
    const resizeObserver = new ResizeObserver(() => {
      this.onResize$.next();
    });
    resizeObserver.observe(ref.nativeElement);
    this.ngUnsubscribe$.subscribe(() => {
      resizeObserver.unobserve(ref.nativeElement);
    });
  }

  ngOnInit() {
    this.onResize$
      .pipe(
        // The first delivery can include layout changes made after view init.
        // Consumers that measure before layout settles need that delivery too.
        skip(this.emitInitialResize ? 0 : 1),
        debounceTime(this.resizeEventDebouncePeriodInMs),
        takeUntil(this.ngUnsubscribe$)
      )
      .subscribe(() => {
        this.onResize.emit();
      });
  }

  ngOnDestroy() {
    this.ngUnsubscribe$.next();
    this.ngUnsubscribe$.complete();
  }
}

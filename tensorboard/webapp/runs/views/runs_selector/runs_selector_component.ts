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
  AfterViewInit,
  ChangeDetectorRef,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  HostListener,
  Input,
  OnDestroy,
} from '@angular/core';
import {RunsTableColumn} from '../runs_table/types';

@Component({
  standalone: false,
  selector: 'runs-selector-component',
  template: `
    <runs-table
      [columns]="columns"
      [experimentIds]="experimentIds"
      [scrollTop]="scrollTop"
      [viewportHeight]="viewportHeight"
      (scrollReset)="resetScroll()"
    ></runs-table>
  `,
  styles: [
    `
      :host {
        display: block;
        height: 100%;
        width: 100%;
        overflow: auto;
      }

      runs-table {
        height: 100%;
      }
    `,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class RunsSelectorComponent implements AfterViewInit, OnDestroy {
  @Input() experimentIds!: string[];
  @Input() columns!: RunsTableColumn[];

  scrollTop = 0;
  viewportHeight = 0;

  private resizeObserver?: ResizeObserver;

  constructor(
    private readonly elementRef: ElementRef<HTMLElement>,
    private readonly changeDetectorRef: ChangeDetectorRef
  ) {}

  ngAfterViewInit() {
    this.resizeObserver = new ResizeObserver(() => {
      this.viewportHeight = this.elementRef.nativeElement.clientHeight;
      this.changeDetectorRef.markForCheck();
    });
    this.resizeObserver.observe(this.elementRef.nativeElement);
  }

  ngOnDestroy() {
    this.resizeObserver?.disconnect();
  }

  resetScroll() {
    this.elementRef.nativeElement.scrollTop = 0;
    this.scrollTop = 0;
  }

  @HostListener('scroll', ['$event'])
  onScroll(event: Event) {
    const element = event.currentTarget as HTMLElement;
    this.scrollTop = element.scrollTop;
    this.viewportHeight = element.clientHeight;
  }
}

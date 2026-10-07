/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

import {holdInViewport, isMouseEventInElement} from './dom';

describe('dom utils', () => {
  describe('isMouseEventInElement', () => {
    [
      {
        testDesc: 'click is to the left of element',
        clientX: 99,
        clientY: 150,
      },
      {
        testDesc: 'click is to the right of element',
        clientX: 201,
        clientY: 150,
      },
      {
        testDesc: 'click is above element',
        clientX: 150,
        clientY: 99,
      },
      {
        testDesc: 'click is below element',
        clientX: 150,
        clientY: 201,
      },
    ].forEach(({testDesc, clientX, clientY}) => {
      it(`returns false when ${testDesc}`, () => {
        const event = new MouseEvent('mouseup', {clientX, clientY});
        const element = document.createElement('div');
        spyOn(element, 'getBoundingClientRect').and.returnValue(
          new DOMRect(100, 100, 100, 100)
        );

        const result = isMouseEventInElement(event, element);

        expect(result).toBeFalse();
      });
    });

    it('returns true when click is within element bounds', () => {
      const event = new MouseEvent('mouseup', {clientX: 150, clientY: 150});
      const element = document.createElement('div');
      spyOn(element, 'getBoundingClientRect').and.returnValue(
        new DOMRect(100, 100, 100, 100)
      );

      const result = isMouseEventInElement(event, element);

      expect(result).toBeTrue();
    });
  });

  describe('holdInViewport', () => {
    let scroller: HTMLElement;
    let above: HTMLElement;
    let target: HTMLElement;
    let release: () => void;

    function block(height: number) {
      const element = document.createElement('div');
      element.style.height = `${height}px`;
      scroller.appendChild(element);
      return element;
    }

    function offset() {
      return (
        target.getBoundingClientRect().top -
        scroller.getBoundingClientRect().top
      );
    }

    // Resize observations are delivered between a frame's animation callbacks
    // and its paint, so they have run by the following frame's callbacks.
    function nextFrame() {
      return new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      });
    }

    beforeEach(() => {
      scroller = document.createElement('div');
      // Disable the browser's own anchoring: it would otherwise hold whatever
      // is topmost and hide whether the target is being held.
      scroller.style.cssText =
        'height: 100px; overflow-y: scroll; overflow-anchor: none;';
      document.body.appendChild(scroller);
      above = block(300);
      target = block(20);
      block(1000);
      scroller.scrollTop = 250;
      release = holdInViewport(scroller, target);
    });

    afterEach(() => {
      release();
      scroller.remove();
    });

    it('keeps the target in place while content above it resizes', async () => {
      expect(offset()).toBe(50);

      above.style.height = '500px';
      await nextFrame();
      expect(offset()).toBe(50);

      above.style.height = '260px';
      await nextFrame();
      expect(offset()).toBe(50);
    });

    for (const type of ['wheel', 'touchstart', 'keydown', 'pointerdown']) {
      it(`lets go of the target on ${type}`, async () => {
        scroller.dispatchEvent(new Event(type));

        above.style.height = '500px';
        await nextFrame();
        expect(offset()).toBe(250);
      });
    }

    it('lets go of the target on input outside the scroller', async () => {
      // A key scrolls the scroller from wherever focus is.
      document.body.dispatchEvent(new Event('keydown'));

      above.style.height = '500px';
      await nextFrame();
      expect(offset()).toBe(250);
    });

    it('lets go of a target that was hidden', async () => {
      target.style.display = 'none';
      above.style.height = '500px';
      await nextFrame();
      expect(scroller.scrollTop).toBe(250);

      // It is not picked up again where it reappears.
      target.style.display = '';
      above.style.height = '700px';
      await nextFrame();
      expect(scroller.scrollTop).toBe(250);
    });

    it('lets go of the target when released', async () => {
      release();

      above.style.height = '500px';
      await nextFrame();
      expect(offset()).toBe(250);
    });

    it('lets go of a target that left the document', async () => {
      target.remove();

      above.style.height = '500px';
      await nextFrame();
      expect(scroller.scrollTop).toBe(250);
    });
  });
});

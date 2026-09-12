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
  ChangeDetectionStrategy,
  Component,
  Input,
  NO_ERRORS_SCHEMA,
} from '@angular/core';
import {ComponentFixture, TestBed} from '@angular/core/testing';
import {By} from '@angular/platform-browser';
import {NoopAnimationsModule} from '@angular/platform-browser/animations';
import {Action, Store} from '@ngrx/store';
import {MockStore, provideMockStore} from '@ngrx/store/testing';
import {State} from '../../../app_state';
import * as actions from '../../actions';
import {reducers} from '../../store/metrics_reducers';
import {appStateFromMetricsState, buildMetricsState} from '../../testing';
import {CardId} from '../../types';
import {CardLazyLoader, CardObserver} from '../card_renderer/card_lazy_loader';

@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'card-view',
  template: `{{ cardId }}`,
})
class TestableCard {
  @Input() cardId!: CardId;
}

interface TestableCardConfig {
  cardId: CardId;
  visible: boolean;
}

@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'testable-cards',
  template: `
    <ng-container *ngFor="let config of configs">
      <card-view
        *ngIf="config.visible"
        [cardId]="config.cardId"
        [cardLazyLoader]="config.cardId"
      ></card-view>
    </ng-container>
  `,
})
class TestableCards {
  @Input() configs!: TestableCardConfig[];
}

describe('card view test', () => {
  let store: MockStore<State>;
  let dispatchedActions: Action[] = [];
  let observeSpy: jasmine.Spy;
  let unobserveSpy: jasmine.Spy;

  function buildIntersectionObserverEntry(
    override: Partial<IntersectionObserverEntry> & {target: Element}
  ): IntersectionObserverEntry {
    return {
      time: 0,
      isIntersecting: false,
      boundingClientRect: new DOMRectReadOnly(),
      intersectionRatio: 0,
      intersectionRect: override.isIntersecting
        ? new DOMRectReadOnly(0, 0, 1, 1)
        : new DOMRectReadOnly(),
      rootBounds: new DOMRectReadOnly(),
      ...override,
    };
  }

  function getCardLazyLoaders(
    fixture: ComponentFixture<TestableCards>
  ): CardLazyLoader[] {
    const cardDebugElements = fixture.debugElement.queryAll(
      By.css('card-view')
    );
    return cardDebugElements.map((debugElement) => {
      return debugElement.injector.get(CardLazyLoader);
    });
  }

  function simulateIntersection(
    cardObserver: CardObserver,
    entries: Array<Partial<IntersectionObserverEntry> & {target: Element}>
  ) {
    cardObserver.onCardIntersectionForTest(
      entries.map(buildIntersectionObserverEntry)
    );
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [NoopAnimationsModule],
      declarations: [CardLazyLoader, TestableCard, TestableCards],
      providers: [
        provideMockStore({
          initialState: appStateFromMetricsState(buildMetricsState()),
        }),
      ],
      schemas: [NO_ERRORS_SCHEMA],
    }).compileComponents();

    dispatchedActions = [];
    store = TestBed.inject<Store<State>>(Store) as MockStore<State>;
    // Cast to jasmine.Spy for compatibility between NgRx dispatch signature overloads.
    (spyOn(store, 'dispatch') as jasmine.Spy).and.callFake((action: Action) => {
      dispatchedActions.push(action);
    });

    observeSpy = spyOn(IntersectionObserver.prototype, 'observe');
    unobserveSpy = spyOn(IntersectionObserver.prototype, 'unobserve');
  });

  it('tracks card removal', () => {
    const fixture = TestBed.createComponent(TestableCards);
    fixture.componentInstance.configs = [
      {cardId: 'card1', visible: true},
    ] as TestableCardConfig[];
    fixture.detectChanges();

    expect(observeSpy).toHaveBeenCalled();

    const directives = getCardLazyLoaders(fixture);
    const cardObserver = directives[0].cardObserver!;

    // Destroy the element. Cleanup should not wait for a later observer event:
    // IntersectionObserver is asynchronous and may never report an exit for
    // paginated or otherwise removed DOM.
    fixture.componentInstance.configs = [
      {cardId: 'card1', visible: false},
    ] as TestableCardConfig[];
    fixture.detectChanges();

    expect(unobserveSpy).toHaveBeenCalled();
    expect(dispatchedActions).toEqual([
      actions.cardVisibilityChanged({
        enteredCards: [],
        exitedCards: [{elementId: jasmine.any(Symbol) as any, cardId: 'card1'}],
      }),
    ]);

    // Simulate a pending 'isIntersecting' event.
    simulateIntersection(cardObserver, [
      {
        time: 10,
        target: directives[0].hostForTest().nativeElement,
        isIntersecting: true,
      },
    ]);

    expect(unobserveSpy).toHaveBeenCalled();
    expect(dispatchedActions).toEqual([
      actions.cardVisibilityChanged({
        enteredCards: [],
        exitedCards: [{elementId: jasmine.any(Symbol) as any, cardId: 'card1'}],
      }),
    ]);

    // Simulate the exiting event. It should also be ignored because the target
    // was already removed.
    simulateIntersection(cardObserver, [
      {
        time: 20,
        target: directives[0].hostForTest().nativeElement,
        isIntersecting: false,
      },
    ]);

    expect(unobserveSpy).toHaveBeenCalled();
    expect(dispatchedActions).toEqual([
      actions.cardVisibilityChanged({
        enteredCards: [],
        exitedCards: [{elementId: jasmine.any(Symbol) as any, cardId: 'card1'}],
      }),
    ]);
  });

  it('prepares nearby cards and updates the buffer when the scroll root resizes', (done) => {
    observeSpy.and.callThrough();
    unobserveSpy.and.callThrough();
    const root = document.createElement('div');
    root.style.cssText =
      'position:fixed;top:0;left:0;width:100px;height:200px;overflow:auto';
    const content = document.createElement('div');
    content.style.cssText = 'position:relative;height:1000px';
    const card = document.createElement('div');
    card.style.cssText = 'position:absolute;top:300px;width:50px;height:40px';
    content.appendChild(card);
    root.appendChild(content);
    document.body.appendChild(root);
    const observer = new CardObserver(root, 1);
    let entered = false;
    observer.initialize((entries, exits) => {
      try {
        if (entries.has(card)) {
          expect(card.getBoundingClientRect().top).toBeGreaterThan(
            root.getBoundingClientRect().bottom
          );
          entered = true;
          root.style.height = '100px';
        } else if (exits.has(card)) {
          expect(entered).toBeTrue();
          expect(card.getBoundingClientRect().height).toBe(40);
          observer.destroy();
          root.remove();
          done();
        }
      } catch (error) {
        observer.destroy();
        root.remove();
        done.fail(error as Error);
      }
    });
    observer.add(card);
  });

  it('ignores duplicate ownership notifications after observer recreation', () => {
    const observer = new CardObserver();
    const notify = jasmine.createSpy('ownership changed');
    const card = document.createElement('div');
    observer.initialize(notify);
    observer.add(card);
    simulateIntersection(observer, [{target: card, isIntersecting: false}]);
    expect(notify).not.toHaveBeenCalled();
    simulateIntersection(observer, [{target: card, isIntersecting: true}]);
    simulateIntersection(observer, [{target: card, isIntersecting: true}]);
    expect(notify).toHaveBeenCalledTimes(1);
    simulateIntersection(observer, [{target: card, isIntersecting: false}]);
    simulateIntersection(observer, [{target: card, isIntersecting: false}]);
    expect(notify).toHaveBeenCalledTimes(2);
    observer.destroy();
  });

  it('uses the newest intersection when queued transitions arrive out of order', () => {
    const observer = new CardObserver();
    const owned = new Set<Element>();
    const card = document.createElement('div');
    observer.initialize((entered, exited) => {
      entered.forEach((target) => owned.add(target));
      exited.forEach((target) => owned.delete(target));
    });
    observer.add(card);
    simulateIntersection(observer, [
      {target: card, time: 20, isIntersecting: true},
      {target: card, time: 10, isIntersecting: false},
    ]);
    expect([...owned]).toEqual([card]);
    simulateIntersection(observer, [
      {target: card, time: 40, isIntersecting: false},
      {target: card, time: 30, isIntersecting: true},
    ]);
    expect([...owned]).toEqual([]);
    observer.destroy();
  });

  it('retains prepared cards across boundary reversals without applying stale exits', () => {
    const observer = new CardObserver();
    const owned = new Set<Element>();
    const target = document.createElement('div');
    observer.initialize((entered, exited) => {
      entered.forEach((element) => owned.add(element));
      exited.forEach((element) => owned.delete(element));
    });
    observer.add(target);
    const notify = (
      mode: 'enter' | 'exit',
      time: number,
      isIntersecting: boolean
    ) =>
      observer.onCardIntersectionForTest(
        [buildIntersectionObserverEntry({target, time, isIntersecting})],
        mode
      );
    notify('enter', 20, true);
    notify('enter', 30, false);
    expect([...owned]).toEqual([target]);
    observer.onCardIntersectionForTest(
      [
        buildIntersectionObserverEntry({
          target,
          time: 40,
          isIntersecting: false,
        }),
        buildIntersectionObserverEntry({
          target,
          time: 50,
          isIntersecting: true,
        }),
      ],
      'exit'
    );
    expect([...owned]).toEqual([target]);
    notify('enter', 70, true);
    notify('exit', 60, false);
    expect([...owned]).toEqual([target]);
    notify('exit', 80, false);
    expect([...owned]).toEqual([]);
    observer.destroy();
  });

  it('owns history only when the intersection contains visible pixels', () => {
    let metricsState = buildMetricsState();
    (store.dispatch as jasmine.Spy).and.callFake((action: Action) => {
      metricsState = reducers(metricsState, action);
    });
    const fixture = TestBed.createComponent(TestableCards);
    fixture.componentInstance.configs = [{cardId: 'card1', visible: true}];
    fixture.detectChanges();
    const directive = getCardLazyLoaders(fixture)[0];
    const observer = directive.cardObserver!;
    const target = directive.hostForTest().nativeElement;

    simulateIntersection(observer, [
      {
        target,
        time: 1,
        isIntersecting: true,
        intersectionRect: new DOMRectReadOnly(0, 0, 10, 0),
      },
    ]);
    expect([...metricsState.visibleCardMap.values()]).toEqual([]);
    simulateIntersection(observer, [{target, time: 2, isIntersecting: true}]);
    expect([...metricsState.visibleCardMap.values()]).toEqual(['card1']);
    simulateIntersection(observer, [
      {
        target,
        time: 3,
        isIntersecting: true,
        intersectionRect: new DOMRectReadOnly(0, 0, 0, 10),
      },
    ]);
    expect([...metricsState.visibleCardMap.values()]).toEqual([]);
    fixture.destroy();
  });
});

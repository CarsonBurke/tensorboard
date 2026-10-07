/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
import {ScrollingModule} from '@angular/cdk/scrolling';
import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  Input,
  NO_ERRORS_SCHEMA,
  Output,
} from '@angular/core';
import {
  ComponentFixture,
  discardPeriodicTasks,
  fakeAsync,
  TestBed,
  tick,
} from '@angular/core/testing';
import {By} from '@angular/platform-browser';
import {NoopAnimationsModule} from '@angular/platform-browser/animations';
import {Action, Store} from '@ngrx/store';
import {MockStore} from '@ngrx/store/testing';
import {State} from '../../../app_state';
import * as selectors from '../../../selectors';
import {
  getCardStateMap,
  getMetricsCardMinWidth,
  getMetricsTagGroupExpansionState,
  getMetricsTagGroupPageIndex,
} from '../../../selectors';
import {selectors as settingsSelectors} from '../../../settings';
import {provideMockTbStore} from '../../../testing/utils';
import * as actions from '../../actions';
import {PluginType} from '../../data_source';
import {CardIdWithMetadata} from '../metrics_view_types';
import {CardGridComponent} from './card_grid_component';
import {CardGridContainer} from './card_grid_container';

const scrollElementHeight = 100;

@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'testable-scrolling-container',
  template: `
    <div cdkScrollable>
      <div class="placeholder">placeholder</div>
      <metrics-card-grid
        [cardIdsWithMetadata]="cardIdsWithMetadata"
        [cardObserver]="cardObserver"
        [groupName]="groupName"
        [serverTotalCards]="serverTotalCards"
      ></metrics-card-grid>
      <div class="placeholder">placeholder</div>
    </div>
  `,
  styles: [
    `
      div {
        position: fixed;
        height: ${scrollElementHeight}px;
        overflow-y: scroll;
        /* Only the grid's own scroll handling is under test. */
        overflow-anchor: none;
      }
      .placeholder {
        position: relative;
        height: 700px;
      }
      metrics-card-grid {
        display: block;
      }
    `,
  ],
})
class TestableScrollingContainer {
  @Input() cardIdsWithMetadata: CardIdWithMetadata[] = [];
  @Input() groupName: string | null = null;
  @Input() serverTotalCards: number | null = null;
}

// Resize observations are delivered between a frame's animation callbacks and
// its paint, so they have run by the following frame's callbacks.
function nextFrame() {
  return new Promise<void>((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  });
}

/**
 * Stub 'card-view' component for ease of testing.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'card-view',
})
class TestableCardView {
  @Output() fullHeightChanged = new EventEmitter<boolean>();
  @Output() fullWidthChanged = new EventEmitter<boolean>();
}

describe('card grid', () => {
  let store: MockStore<State>;
  let dispatchedActions: Action[];
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [NoopAnimationsModule, ScrollingModule],
      declarations: [
        CardGridComponent,
        CardGridContainer,
        TestableCardView,
        TestableScrollingContainer,
      ],
      providers: [provideMockTbStore()],
      schemas: [NO_ERRORS_SCHEMA],
    }).compileComponents();

    store = TestBed.inject<Store<State>>(Store) as MockStore<State>;
    dispatchedActions = [];
    (spyOn(store, 'dispatch') as jasmine.Spy).and.callFake((action: Action) => {
      dispatchedActions.push(action);
    });
    store.overrideSelector(selectors.getRunColorMap, {});
    store.overrideSelector(getMetricsTagGroupExpansionState, true);
    store.overrideSelector(getMetricsTagGroupPageIndex, 0);
    store.overrideSelector(getMetricsCardMinWidth, 30);
    store.overrideSelector(settingsSelectors.getPageSize, 10);
    store.overrideSelector(getCardStateMap, {});
  });

  afterEach(() => {
    store?.resetSelectors();
  });

  it('keeps pagination button position when page size changes', async () => {
    store.overrideSelector(settingsSelectors.getPageSize, 2);
    let scrollOffset = 30;
    const fixture = TestBed.createComponent(TestableScrollingContainer);
    // With 3 cards and a page size of 2 the number of cards on a page changes
    // from 2 to 1 when going from the first to second page. This is crucial for
    // this test.
    fixture.componentInstance.cardIdsWithMetadata = [
      {
        cardId: 'card1',
        plugin: PluginType.SCALARS,
        tag: 'tagA',
        runId: null,
      },
      {
        cardId: 'card2',
        plugin: PluginType.SCALARS,
        tag: 'tagA/Images',
        runId: 'run1',
        sample: 0,
      },
      {
        cardId: 'card3',
        plugin: PluginType.SCALARS,
        tag: 'tagB/meow/cat',
        runId: 'run1',
        sample: 0,
      },
    ];
    fixture.detectChanges();
    const [topNextButtons, bottomNextButtons] = fixture.debugElement
      .queryAll(By.css('.next'))
      .map((nextDebugElements) => {
        return nextDebugElements.nativeElement!;
      });
    const [topPreviousButtons, bottomPreviousButtons] = fixture.debugElement
      .queryAll(By.css('.prev'))
      .map((nextDebugElements) => {
        return nextDebugElements.nativeElement!;
      });
    const PaginationInput: HTMLInputElement = fixture.debugElement.query(
      By.css('input')
    ).nativeElement;
    const scrollingElement = fixture.nativeElement.children[0];

    // Test scrolling adjustments on bottom next button.
    scrollingElement.scrollTo(0, bottomNextButtons.offsetTop - scrollOffset);
    bottomNextButtons.click();
    fixture.detectChanges();
    // To ensure the click did change the size of the CardGrid ensure make sure
    // the button has moved.
    expect(
      bottomNextButtons.offsetTop - scrollingElement.scrollTop
    ).not.toEqual(scrollOffset);
    await nextFrame();
    expect(bottomNextButtons.offsetTop - scrollingElement.scrollTop).toEqual(
      scrollOffset
    );

    // Test scrolling adjustments on top previous button.
    scrollingElement.scrollTo(0, topPreviousButtons.offsetTop - scrollOffset);
    topPreviousButtons.click();
    fixture.detectChanges();
    await nextFrame();
    expect(topPreviousButtons.offsetTop - scrollingElement.scrollTop).toEqual(
      scrollOffset
    );

    // Test scrolling adjustments on top next button.
    scrollingElement.scrollTo(0, topNextButtons.offsetTop - scrollOffset);
    topNextButtons.click();
    fixture.detectChanges();
    await nextFrame();
    expect(topNextButtons.offsetTop - scrollingElement.scrollTop).toEqual(
      scrollOffset
    );

    // Test scrolling adjustments on bottom previous button.
    scrollingElement.scrollTo(
      0,
      bottomPreviousButtons.offsetTop - scrollOffset
    );
    bottomPreviousButtons.click();
    fixture.detectChanges();
    // To ensure the click did change the size of the CardGrid ensure make sure
    // the button has moved.
    expect(
      bottomPreviousButtons.offsetTop - scrollingElement.scrollTop
    ).not.toEqual(scrollOffset);
    await nextFrame();
    expect(
      bottomPreviousButtons.offsetTop - scrollingElement.scrollTop
    ).toEqual(scrollOffset);

    // Test changes to input.
    scrollingElement.scrollTo(0, PaginationInput.offsetTop - scrollOffset);
    PaginationInput.value = '2';
    PaginationInput.dispatchEvent(new Event('input'));
    fixture.detectChanges();
    // To ensure the click did change the size of the CardGrid ensure make sure
    // the next button has moved.
    expect(PaginationInput.offsetTop - scrollingElement.scrollTop).not.toEqual(
      scrollOffset
    );
    await nextFrame();
    expect(PaginationInput.offsetTop - scrollingElement.scrollTop).toEqual(
      scrollOffset
    );
  });

  it('keeps holding a pagination button while its page keeps resizing', async () => {
    store.overrideSelector(settingsSelectors.getPageSize, 2);
    const fixture = TestBed.createComponent(TestableScrollingContainer);
    fixture.componentInstance.cardIdsWithMetadata = [
      'card1',
      'card2',
      'card3',
    ].map((cardId) => ({
      cardId,
      plugin: PluginType.SCALARS,
      tag: cardId,
      runId: null,
    }));
    fixture.detectChanges();
    const scrollingElement: HTMLElement = fixture.nativeElement.children[0];
    const bottomNext: HTMLElement = fixture.debugElement.queryAll(
      By.css('.next')
    )[1].nativeElement;
    const offset = () => bottomNext.offsetTop - scrollingElement.scrollTop;
    scrollingElement.scrollTo(0, bottomNext.offsetTop - 30);

    bottomNext.click();
    fixture.detectChanges();
    await nextFrame();
    expect(offset()).toBe(30);

    // The newly paged-in card loads and grows after the page was rendered.
    const card: HTMLElement = fixture.debugElement.query(
      By.css('card-view')
    ).nativeElement;
    card.style.height = '900px';
    await nextFrame();
    expect(offset()).toBe(30);

    // Scrolling away is the reader's call; the button is no longer held.
    scrollingElement.dispatchEvent(new Event('wheel'));
    card.style.height = '1200px';
    await nextFrame();
    expect(offset()).toBe(330);
  });

  it('does not hold the page input when it reports the current page', async () => {
    store.overrideSelector(settingsSelectors.getPageSize, 2);
    const fixture = TestBed.createComponent(TestableScrollingContainer);
    fixture.componentInstance.cardIdsWithMetadata = [
      'card1',
      'card2',
      'card3',
    ].map((cardId) => ({
      cardId,
      plugin: PluginType.SCALARS,
      tag: cardId,
      runId: null,
    }));
    fixture.detectChanges();
    const scrollingElement: HTMLElement = fixture.nativeElement.children[0];
    const input: HTMLInputElement = fixture.debugElement.query(
      By.css('input')
    ).nativeElement;
    const offset = () => input.offsetTop - scrollingElement.scrollTop;
    scrollingElement.scrollTo(0, input.offsetTop - 30);

    // Fired when the input loses focus, e.g. to a click that is about to
    // resize content above it. That click's target must not be scrolled away.
    input.dispatchEvent(new Event('change'));
    fixture.detectChanges();
    const card: HTMLElement = fixture.debugElement.query(
      By.css('card-view')
    ).nativeElement;
    card.style.height = '900px';
    await nextFrame();

    expect(offset()).toBeGreaterThan(30);
  });

  describe('pending server pages', () => {
    function createGrid(totalCards: number | null, cardIds: string[] = []) {
      const fixture = TestBed.createComponent(TestableScrollingContainer);
      fixture.componentInstance.groupName = 'tagA';
      fixture.componentInstance.serverTotalCards = totalCards;
      fixture.componentInstance.cardIdsWithMetadata = cardIds.map((cardId) => ({
        cardId,
        plugin: PluginType.SCALARS,
        tag: `tagA/${cardId}`,
        runId: null,
      }));
      fixture.detectChanges();
      return fixture;
    }

    function getPlaceholders(
      fixture: ComponentFixture<TestableScrollingContainer>
    ) {
      return fixture.debugElement.queryAll(By.css('.card-placeholder'));
    }

    it('reserves a slot for every card of a page that is not listed yet', () => {
      const fixture = createGrid(25);
      expect(getPlaceholders(fixture).length).toBe(10);
      expect(fixture.debugElement.queryAll(By.css('card-view')).length).toBe(0);
    });

    it('reserves only the remaining cards on the last page', () => {
      store.overrideSelector(getMetricsTagGroupPageIndex, 2);
      const fixture = createGrid(25);
      expect(getPlaceholders(fixture).length).toBe(5);
    });

    it('replaces the slots with the cards once they are listed', () => {
      const fixture = createGrid(25, ['card1', 'card2']);
      expect(getPlaceholders(fixture).length).toBe(0);
      expect(fixture.debugElement.queryAll(By.css('card-view')).length).toBe(2);
    });

    it('reserves nothing for a group without cards', () => {
      const fixture = createGrid(0);
      expect(getPlaceholders(fixture).length).toBe(0);
    });

    it('reserves nothing for a collapsed group', () => {
      store.overrideSelector(getMetricsTagGroupExpansionState, false);
      const fixture = createGrid(25);
      expect(getPlaceholders(fixture).length).toBe(0);
    });

    it('reserves nothing when the client holds the whole card list', () => {
      const fixture = createGrid(null);
      expect(getPlaceholders(fixture).length).toBe(0);
    });
  });

  it('dispatches page index changes for grouped grids', fakeAsync(() => {
    store.overrideSelector(settingsSelectors.getPageSize, 1);
    store.overrideSelector(getMetricsTagGroupPageIndex, 0);
    const fixture = TestBed.createComponent(TestableScrollingContainer);
    fixture.componentInstance.groupName = 'tagA';
    fixture.componentInstance.cardIdsWithMetadata = [
      {
        cardId: 'card1',
        plugin: PluginType.SCALARS,
        tag: 'tagA/one',
        runId: null,
      },
      {
        cardId: 'card2',
        plugin: PluginType.SCALARS,
        tag: 'tagA/two',
        runId: null,
      },
    ];
    fixture.detectChanges();

    fixture.debugElement.query(By.css('.next')).nativeElement.click();
    tick(0);

    expect(dispatchedActions).toContain(
      actions.metricsTagGroupPageIndexChanged({
        tagGroup: 'tagA',
        pageIndex: 1,
      })
    );
    discardPeriodicTasks();
  }));

  describe('card dimensions', () => {
    let fixture: ComponentFixture<TestableScrollingContainer>;

    function createComponent() {
      fixture = TestBed.createComponent(TestableScrollingContainer);
      fixture.componentInstance.cardIdsWithMetadata = [
        {
          cardId: 'card1',
          plugin: PluginType.SCALARS,
          tag: 'tagA',
          runId: null,
        },
        {
          cardId: 'card2',
          plugin: PluginType.SCALARS,
          tag: 'tagA',
          runId: null,
        },
        {
          cardId: 'card3',
          plugin: PluginType.SCALARS,
          tag: 'tagA',
          runId: null,
        },
      ];
      fixture.detectChanges();

      return fixture;
    }

    it('shows cards at min dimensions by default', () => {
      const fixture = createComponent();
      const cardSpaces = fixture.debugElement.queryAll(By.css('.card-space'));
      expect(cardSpaces[0].nativeElement.classList).not.toContain(
        'full-height'
      );
      expect(cardSpaces[0].nativeElement.classList).not.toContain('full-width');
      expect(cardSpaces[1].nativeElement.classList).not.toContain(
        'full-height'
      );
      expect(cardSpaces[1].nativeElement.classList).not.toContain('full-width');
      expect(cardSpaces[2].nativeElement.classList).not.toContain(
        'full-height'
      );
      expect(cardSpaces[2].nativeElement.classList).not.toContain('full-width');
    });

    it('changes height after card event', () => {
      const fixture = createComponent();
      const cardViews = fixture.debugElement.queryAll(By.css('card-view'));
      const cardSpaces = fixture.debugElement.queryAll(By.css('.card-space'));

      cardViews[1].componentInstance.fullHeightChanged.emit(true);
      fixture.detectChanges();
      expect(cardSpaces[0].nativeElement.classList).not.toContain(
        'full-height'
      );
      expect(cardSpaces[1].nativeElement.classList).toContain('full-height');
      expect(cardSpaces[2].nativeElement.classList).not.toContain(
        'full-height'
      );

      cardViews[0].componentInstance.fullHeightChanged.emit(true);
      fixture.detectChanges();
      expect(cardSpaces[0].nativeElement.classList).toContain('full-height');
      expect(cardSpaces[1].nativeElement.classList).toContain('full-height');
      expect(cardSpaces[2].nativeElement.classList).not.toContain(
        'full-height'
      );

      cardViews[1].componentInstance.fullHeightChanged.emit(false);
      fixture.detectChanges();
      expect(cardSpaces[0].nativeElement.classList).toContain('full-height');
      expect(cardSpaces[1].nativeElement.classList).not.toContain(
        'full-height'
      );
      expect(cardSpaces[2].nativeElement.classList).not.toContain(
        'full-height'
      );

      cardViews[0].componentInstance.fullHeightChanged.emit(false);
      fixture.detectChanges();
      expect(cardSpaces[0].nativeElement.classList).not.toContain(
        'full-height'
      );
      expect(cardSpaces[1].nativeElement.classList).not.toContain(
        'full-height'
      );
      expect(cardSpaces[2].nativeElement.classList).not.toContain(
        'full-height'
      );
    });

    it('does not change height if emitted value is same', () => {
      const fixture = createComponent();
      const cardViews = fixture.debugElement.queryAll(By.css('card-view'));
      const cardSpaces = fixture.debugElement.queryAll(By.css('.card-space'));

      cardViews[1].componentInstance.fullHeightChanged.emit(true);
      fixture.detectChanges();
      expect(cardSpaces[0].nativeElement.classList).not.toContain(
        'full-height'
      );
      expect(cardSpaces[1].nativeElement.classList).toContain('full-height');
      expect(cardSpaces[2].nativeElement.classList).not.toContain(
        'full-height'
      );

      cardViews[0].componentInstance.fullHeightChanged.emit(false);
      fixture.detectChanges();
      expect(cardSpaces[0].nativeElement.classList).not.toContain(
        'full-height'
      );
      expect(cardSpaces[1].nativeElement.classList).toContain('full-height');
      expect(cardSpaces[2].nativeElement.classList).not.toContain(
        'full-height'
      );

      cardViews[1].componentInstance.fullHeightChanged.emit(true);
      fixture.detectChanges();
      expect(cardSpaces[0].nativeElement.classList).not.toContain(
        'full-height'
      );
      expect(cardSpaces[1].nativeElement.classList).toContain('full-height');
      expect(cardSpaces[2].nativeElement.classList).not.toContain(
        'full-height'
      );
    });

    it('renders card width based on card state table expanded', () => {
      store.overrideSelector(getCardStateMap, {card2: {tableExpanded: true}});
      let fixture = createComponent();
      let cardSpaces = fixture.debugElement.queryAll(By.css('.card-space'));
      expect(cardSpaces[0].nativeElement.classList).not.toContain(
        'full-height'
      );
      expect(cardSpaces[1].nativeElement.classList).toContain('full-height');
      expect(cardSpaces[2].nativeElement.classList).not.toContain(
        'full-height'
      );

      store.overrideSelector(getCardStateMap, {
        card1: {tableExpanded: true},
        card2: {tableExpanded: true},
      });
      fixture = createComponent();
      cardSpaces = fixture.debugElement.queryAll(By.css('.card-space'));
      expect(cardSpaces[0].nativeElement.classList).toContain('full-height');
      expect(cardSpaces[1].nativeElement.classList).toContain('full-height');
      expect(cardSpaces[2].nativeElement.classList).not.toContain(
        'full-height'
      );

      store.overrideSelector(getCardStateMap, {
        card1: {tableExpanded: false},
        card2: {tableExpanded: true},
      });
      fixture = createComponent();
      cardSpaces = fixture.debugElement.queryAll(By.css('.card-space'));
      expect(cardSpaces[0].nativeElement.classList).not.toContain(
        'full-height'
      );
      expect(cardSpaces[1].nativeElement.classList).toContain('full-height');
      expect(cardSpaces[2].nativeElement.classList).not.toContain(
        'full-height'
      );

      store.overrideSelector(getCardStateMap, {});
      fixture = createComponent();
      cardSpaces = fixture.debugElement.queryAll(By.css('.card-space'));
      expect(cardSpaces[0].nativeElement.classList).not.toContain(
        'full-height'
      );
      expect(cardSpaces[1].nativeElement.classList).not.toContain(
        'full-height'
      );
      expect(cardSpaces[2].nativeElement.classList).not.toContain(
        'full-height'
      );
    });

    it('renders card width based on card state full width', () => {
      store.overrideSelector(getCardStateMap, {card3: {fullWidth: true}});
      let fixture = createComponent();
      let cardSpaces = fixture.debugElement.queryAll(By.css('.card-space'));
      expect(cardSpaces[0].nativeElement.classList).not.toContain('full-width');
      expect(cardSpaces[1].nativeElement.classList).not.toContain('full-width');
      expect(cardSpaces[2].nativeElement.classList).toContain('full-width');

      store.overrideSelector(getCardStateMap, {
        card2: {fullWidth: true},
        card3: {fullWidth: true},
      });
      fixture = createComponent();
      cardSpaces = fixture.debugElement.queryAll(By.css('.card-space'));
      expect(cardSpaces[0].nativeElement.classList).not.toContain('full-width');
      expect(cardSpaces[1].nativeElement.classList).toContain('full-width');
      expect(cardSpaces[2].nativeElement.classList).toContain('full-width');

      store.overrideSelector(getCardStateMap, {
        card2: {fullWidth: false},
        card3: {fullWidth: true},
      });
      fixture = createComponent();
      cardSpaces = fixture.debugElement.queryAll(By.css('.card-space'));
      expect(cardSpaces[0].nativeElement.classList).not.toContain('full-width');
      expect(cardSpaces[1].nativeElement.classList).not.toContain('full-width');
      expect(cardSpaces[2].nativeElement.classList).toContain('full-width');

      store.overrideSelector(getCardStateMap, {});
      fixture = createComponent();
      cardSpaces = fixture.debugElement.queryAll(By.css('.card-space'));
      expect(cardSpaces[0].nativeElement.classList).not.toContain('full-width');
      expect(cardSpaces[1].nativeElement.classList).not.toContain('full-width');
      expect(cardSpaces[2].nativeElement.classList).not.toContain('full-width');
    });
  });
});

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
  EventEmitter,
  Input,
  NO_ERRORS_SCHEMA,
  Output,
} from '@angular/core';
import {TestBed} from '@angular/core/testing';
import {By} from '@angular/platform-browser';
import {NoopAnimationsModule} from '@angular/platform-browser/animations';
import {Action, Store} from '@ngrx/store';
import {MockStore, provideMockStore} from '@ngrx/store/testing';
import {State} from '../../../app_state';
import * as selectors from '../../../selectors';
import {RunColorScale} from '../../../types/ui';
import * as actions from '../../actions';
import {PluginType} from '../../data_source';
import {appStateFromMetricsState, buildMetricsState} from '../../testing';
import {CardViewComponent} from './card_view_component';
import {CardViewContainer} from './card_view_container';

@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'scalar-card',
  template: ``,
})
class TestableScalarCard {
  @Input() runColorScale!: RunColorScale;
  @Output() fullWidthChanged = new EventEmitter<boolean>();
  @Output() fullHeightChanged = new EventEmitter<boolean>();
  @Output() pinStateChanged = new EventEmitter<void>();
}

describe('card view test', () => {
  let store: MockStore<State>;
  let dispatchedActions: Action[] = [];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [NoopAnimationsModule],
      declarations: [CardViewComponent, CardViewContainer, TestableScalarCard],
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
    store.overrideSelector(selectors.getRunColorMap, {});
    store.overrideSelector(selectors.getVisibleCardIdSet, new Set<string>());
  });

  afterEach(() => {
    store?.resetSelectors();
  });

  it('mounts charts only while their card belongs to the render buffer', () => {
    const fixture = TestBed.createComponent(CardViewContainer);
    fixture.componentInstance.cardId = 'cardId';
    fixture.componentInstance.pluginType = PluginType.SCALARS;
    fixture.detectChanges();

    expect(fixture.debugElement.query(By.css('scalar-card'))).toBeNull();
    fixture.detectChanges();

    store.overrideSelector(selectors.getVisibleCardIdSet, new Set(['cardId']));
    store.refreshState();
    fixture.detectChanges();
    expect(fixture.debugElement.query(By.css('scalar-card'))).not.toBeNull();
    store.overrideSelector(selectors.getVisibleCardIdSet, new Set<string>());
    store.refreshState();
    fixture.detectChanges();
    expect(fixture.debugElement.query(By.css('scalar-card'))).toBeNull();
  });

  [
    {tagName: 'scalar-card', pluginType: PluginType.SCALARS},
    {tagName: 'image-card', pluginType: PluginType.IMAGES},
    {tagName: 'histogram-card', pluginType: PluginType.HISTOGRAMS},
  ].forEach(({tagName, pluginType}) => {
    it(`renders proper component for pluginType: ${pluginType}`, () => {
      const fixture = TestBed.createComponent(CardViewContainer);
      fixture.componentInstance.cardId = 'cardId';
      fixture.componentInstance.pluginType = pluginType;
      store.overrideSelector(
        selectors.getVisibleCardIdSet,
        new Set(['cardId'])
      );
      fixture.detectChanges();

      expect(fixture.debugElement.query(By.css(tagName))).not.toBeNull();
    });
  });

  it('dispatches action when pin state changes', () => {
    const fixture = TestBed.createComponent(CardViewContainer);
    fixture.componentInstance.cardId = 'cardId';
    fixture.componentInstance.pluginType = PluginType.SCALARS;
    store.overrideSelector(selectors.getVisibleCardIdSet, new Set(['cardId']));
    fixture.detectChanges();

    const scalarCard = fixture.debugElement.query(By.css('scalar-card'));
    scalarCard.componentInstance.pinStateChanged.emit(true);
    fixture.detectChanges();

    expect(dispatchedActions).toEqual([
      actions.cardPinStateToggled({
        cardId: 'cardId',
        canCreateNewPins: true,
        wasPinned: false,
      }),
    ]);

    store.overrideSelector(selectors.getCardPinnedState, true);
    store.refreshState();
    scalarCard.componentInstance.pinStateChanged.emit(false);
    fixture.detectChanges();

    expect(dispatchedActions).toEqual([
      actions.cardPinStateToggled({
        cardId: 'cardId',
        canCreateNewPins: true,
        wasPinned: false,
      }),
      actions.cardPinStateToggled({
        cardId: 'cardId',
        canCreateNewPins: true,
        wasPinned: true,
      }),
    ]);
  });
});

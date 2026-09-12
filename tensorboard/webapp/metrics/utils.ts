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
import {DeepReadonly} from '../util/types';
import {CardGroup, CardIdWithMetadata} from './types';

export function groupCardIdWithMetdata(
  cards: DeepReadonly<CardIdWithMetadata[]>
): CardGroup[] {
  const tagPrefix = new Map<string, CardGroup>();

  // Callers usually pass a list already ordered by `compareTagNames`. A linear
  // check costs far less than re-sorting thousands of cards.
  let sortedCards: DeepReadonly<CardIdWithMetadata[]> = cards;
  for (let i = 1; i < cards.length; i++) {
    if (compareTagNames(cards[i - 1].tag, cards[i].tag) > 0) {
      sortedCards = cards.slice().sort((cardA, cardB) => {
        return compareTagNames(cardA.tag, cardB.tag);
      });
      break;
    }
  }

  for (const card of sortedCards) {
    const groupName = getTagGroupName(card.tag);

    const group = tagPrefix.get(groupName);
    if (group) {
      group.items.push(card);
    } else {
      tagPrefix.set(groupName, {groupName, items: [card]});
    }
  }

  return [...tagPrefix.values()];
}

export function getTagGroupName(tag: string): string {
  return tag.split('/', 1)[0];
}

// TODO(b/154055328): combine this with the OSS ts_library compat version.
// Adopted from tensorboard/components/vz_sorting/sorting.js
// Delta:
// - better typing
// - human readable variable names
// - removed componentization by "_".

/**
 * Compares tag names asciinumerically broken into components.
 *
 * Unlike the standard asciibetical comparator, this function knows that 'a10b'
 * > 'a2b'. Fixed point and engineering notation are supported. This function
 * also splits the input by slash to perform array comparison. Therefore it
 * knows that 'a/a' < 'a+/a' even though '+' < '/' in the ASCII table.
 */
export function compareTagNames(tagA: string, tagB: string) {
  let aIndex = 0;
  let bIndex = 0;

  while (true) {
    if (aIndex === tagA.length) {
      return bIndex === tagB.length ? 0 : -1;
    }
    if (bIndex === tagB.length) {
      return 1;
    }

    const a = tagA.charCodeAt(aIndex);
    const b = tagB.charCodeAt(bIndex);

    if (isDigit(a) && isDigit(b)) {
      const aNumberStart = aIndex;
      const bNumberStart = bIndex;
      aIndex = consumeNumber(tagA, aIndex + 1);
      bIndex = consumeNumber(tagB, bIndex + 1);
      const an = Number(tagA.slice(aNumberStart, aIndex));
      const bn = Number(tagB.slice(bNumberStart, bIndex));
      if (an < bn) {
        return -1;
      }
      if (an > bn) {
        return 1;
      }
      continue;
    }

    if (isBreak(a)) {
      if (!isBreak(b)) {
        return -1;
      }
    } else if (isBreak(b)) {
      return 1;
    } else if (a < b) {
      return -1;
    } else if (a > b) {
      return 1;
    }

    aIndex++;
    bIndex++;
  }
}

const enum NumberState {
  NATURAL,
  REAL,
  EXPONENT_SIGN,
  EXPONENT,
}

const CHAR_CODE_DOT = 0x2e;
const CHAR_CODE_SLASH = 0x2f;
const CHAR_CODE_PLUS = 0x2b;
const CHAR_CODE_MINUS = 0x2d;
const CHAR_CODE_ZERO = 0x30;
const CHAR_CODE_NINE = 0x39;
const CHAR_CODE_UPPER_E = 0x45;
const CHAR_CODE_LOWER_E = 0x65;

/**
 * Returns endIndex of a number sequence in string starting from startIndex.
 *
 * The method can handle scientific notation, real and natural numbers, and
 * numbers with exponents. Do note that it does not treat decimals that start
 * with "." as a real number.
 */
function consumeNumber(s: string, startIndex: number): number {
  let state = NumberState.NATURAL;
  let i = startIndex;
  for (; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (state === NumberState.NATURAL) {
      if (c === CHAR_CODE_DOT) {
        state = NumberState.REAL;
      } else if (c === CHAR_CODE_LOWER_E || c === CHAR_CODE_UPPER_E) {
        state = NumberState.EXPONENT_SIGN;
      } else if (!isDigit(c)) {
        break;
      }
    } else if (state === NumberState.REAL) {
      if (c === CHAR_CODE_LOWER_E || c === CHAR_CODE_UPPER_E) {
        state = NumberState.EXPONENT_SIGN;
      } else if (!isDigit(c)) {
        break;
      }
    } else if (state === NumberState.EXPONENT_SIGN) {
      if (isDigit(c) || c === CHAR_CODE_PLUS || c === CHAR_CODE_MINUS) {
        state = NumberState.EXPONENT;
      } else {
        break;
      }
    } else if (state === NumberState.EXPONENT) {
      if (!isDigit(c)) {
        break;
      }
    }
  }
  return i;
}

function isDigit(charCode: number): boolean {
  return CHAR_CODE_ZERO <= charCode && charCode <= CHAR_CODE_NINE;
}

function isBreak(charCode: number): boolean {
  return charCode === CHAR_CODE_SLASH || isDigit(charCode);
}

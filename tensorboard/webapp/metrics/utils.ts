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
 * > 'a2b'. Fixed point and engineering notation are supported. Slashes sort
 * before numbers, which sort before every other character, so 'a/a' < 'a+/a'
 * even though '+' < '/' in the ASCII table.
 *
 * Each tag is split into a fixed token sequence (slash, number, or UTF-16 code
 * unit) and the sequences are compared lexicographically, with equal sequences
 * (equivalent number spellings) broken by plain string order. That makes this
 * a strict total order, which the data server relies on: it collates a
 * persisted index with the identical `compare_tag_names` and pages tags in
 * that order. Keep the two in sync.
 */
export function compareTagNames(tagA: string, tagB: string): number {
  let aIndex = 0;
  let bIndex = 0;
  while (true) {
    if (aIndex === tagA.length || bIndex === tagB.length) {
      if (aIndex !== tagA.length) return 1;
      if (bIndex !== tagB.length) return -1;
      return tagA < tagB ? -1 : tagA > tagB ? 1 : 0;
    }
    const a = tagA.charCodeAt(aIndex);
    const b = tagB.charCodeAt(bIndex);
    const aRank = tokenRank(a);
    const bRank = tokenRank(b);
    if (aRank !== bRank) {
      return aRank - bRank;
    }
    if (aRank === TokenRank.NUMBER) {
      const aEnd = consumeNumber(tagA, aIndex);
      const bEnd = consumeNumber(tagB, bIndex);
      const an = Number(tagA.slice(aIndex, aEnd));
      const bn = Number(tagB.slice(bIndex, bEnd));
      if (an !== bn) {
        return an < bn ? -1 : 1;
      }
      aIndex = aEnd;
      bIndex = bEnd;
      continue;
    }
    if (a !== b) {
      return a < b ? -1 : 1;
    }
    aIndex++;
    bIndex++;
  }
}

const enum TokenRank {
  SLASH,
  NUMBER,
  OTHER,
}

function tokenRank(charCode: number): TokenRank {
  if (charCode === CHAR_CODE_SLASH) return TokenRank.SLASH;
  if (isDigit(charCode)) return TokenRank.NUMBER;
  return TokenRank.OTHER;
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
 * Returns the end index of the decimal literal starting at `startIndex`:
 * `digits [. digits*] [(e|E) [+|-] digits+]`. A dangling exponent marker (as
 * in 'res1encoder') is not part of the number, so parsing never yields NaN.
 */
function consumeNumber(s: string, startIndex: number): number {
  const digits = (from: number) => {
    while (from < s.length && isDigit(s.charCodeAt(from))) from++;
    return from;
  };
  let end = digits(startIndex);
  if (s.charCodeAt(end) === CHAR_CODE_DOT) {
    end = digits(end + 1);
  }
  const e = s.charCodeAt(end);
  if (e === CHAR_CODE_LOWER_E || e === CHAR_CODE_UPPER_E) {
    const sign = s.charCodeAt(end + 1);
    const exponent =
      sign === CHAR_CODE_PLUS || sign === CHAR_CODE_MINUS ? end + 2 : end + 1;
    if (isDigit(s.charCodeAt(exponent))) {
      end = digits(exponent);
    }
  }
  return end;
}

function isDigit(charCode: number): boolean {
  return CHAR_CODE_ZERO <= charCode && charCode <= CHAR_CODE_NINE;
}

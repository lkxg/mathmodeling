import unittest

from q1.asr_audit import alternative_matches, classify, compare, edit_counts, normalize_text


class TranscriptAuditTests(unittest.TestCase):
    def test_edit_accounting_for_substitution_deletion_and_insertion(self):
        cases = [(['a', 'b'], ['a', 'c'], (1, 0, 0)),
                 (['a', 'b'], ['a'], (0, 1, 0)),
                 (['a'], ['a', 'b'], (0, 0, 1))]
        for reference, hypothesis, counts in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = edit_counts(reference, hypothesis)
                self.assertEqual(tuple(result[k] for k in ['substitutions', 'deletions', 'insertions']), counts)
                self.assertEqual(result['word_edit_distance'], sum(counts))
                self.assertEqual(result['matches'] + result['substitutions'] + result['deletions'], len(reference))
                self.assertEqual(result['matches'] + result['substitutions'] + result['insertions'], len(hypothesis))

    def test_hallucinated_insertions_can_exceed_one_hundred_percent(self):
        result = edit_counts(['yes'], ['no', 'one', 'said', 'this'])
        self.assertEqual(result['wer'], 4)
        self.assertEqual(result['normalized_word_distance'], 1)

    def test_empty_reference_is_not_perfect_agreement(self):
        self.assertIsNone(edit_counts([], ['hello'])['wer'])
        self.assertEqual(edit_counts(['hello'], [])['deletions'], 1)

    def record(self, **changes):
        return dict({'digital_silence': False, 'asr_text': 'some words', 'language': 'en',
                     'generation_limit': False}, **changes)

    def test_silence_and_non_english_are_not_declared_content_mismatches(self):
        self.assertEqual(classify(self.record(digital_silence=True), {'wer': 1}), 'digital_silence')
        self.assertEqual(classify(self.record(language='he'), {'wer': 1}), 'language_difference')

    def test_unreliable_asr_cannot_certify_agreement(self):
        self.assertEqual(classify(self.record(language=None), {'wer': 0}), 'asr_uncertain')
        self.assertEqual(classify(self.record(generation_limit=True), {'wer': 0}), 'asr_uncertain')
        self.assertEqual(classify(self.record(asr_text=''), {'wer': 0}), 'insufficient_speech')

    def test_comparison_categories_keep_small_differences_separate(self):
        self.assertEqual(classify(self.record(), {'wer': .1}), 'mostly_agree')
        self.assertEqual(classify(self.record(), {'wer': .3}), 'moderate_difference')
        self.assertEqual(classify(self.record(), {'wer': .8}), 'large_difference')

    def test_alternative_clip_match_excludes_self_and_short_generic_phrases(self):
        words = 'this is the actual spoken sentence'.split()
        refs = {'own': words, 'neighbor': words, 'generic': ['thank', 'you']}
        result = alternative_matches('own', words, refs, 1)
        self.assertEqual(result, [{'key': 'neighbor', 'normalized_word_distance': 0}])
        self.assertEqual(alternative_matches('own', ['thank', 'you'], refs, 1), [])

    def test_unsupported_qwen_label_does_not_certify_english_agreement(self):
        source = {'key': 'sample', 'raw_text': 'some words', 'digital_silence': False}
        predictions = {'sample': self.record(language='he')}
        result = compare([source], predictions, ['en', 'ar'])[0]
        self.assertFalse(result['language_supported'])
        self.assertFalse(result['wer_comparable'])
        self.assertEqual(result['status'], 'language_difference')
        self.assertEqual(result['alternative_matches'], [])

    def test_normalization_equates_surface_variations(self):
        pairs = [("[Speaker:] I’m in the U.S., and I can't go.", 'I am in the US and I cannot go'),
                 ('twelve hundred people', '1,200 people'),
                 ('eight hundred and twenty-five', '825'),
                 ('one million two hundred thousand', '1,200,000'),
                 ('a thousand', '1000'), ('tenth', '10th'), ('ＦＩＦＴＹ', '50')]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                self.assertEqual(normalize_text(a), normalize_text(b))

    def test_normalization_preserves_word_repetitions_and_content_differences(self):
        self.assertEqual(normalize_text('one one and two'), '1 1 and 2')
        self.assertNotEqual(normalize_text('he can go'), normalize_text("he can't go"))
        self.assertNotEqual(normalize_text('fourteen'), normalize_text('forty'))
        self.assertEqual(normalize_text('1.25'), '1.25')

    def test_comparison_retains_source_text_and_uses_normalized_words(self):
        source = {'key': 'sample', 'raw_text': 'Twelve hundred.', 'digital_silence': False}
        result = compare([source], {'sample': self.record(asr_text='1,200')}, ['en'])[0]
        self.assertEqual(result['raw_text'], source['raw_text'])
        self.assertEqual(result['asr_text'], '1,200')
        self.assertEqual(result['wer'], 0)
        self.assertTrue(result['language_supported'])
        self.assertEqual(result['status'], 'mostly_agree')


if __name__ == '__main__':
    unittest.main()

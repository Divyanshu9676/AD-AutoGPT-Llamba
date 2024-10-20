import unittest
from unittest.mock import patch, MagicMock
from app import find_files, get_lemma, get_time_difference, scrape_place_text, prepare_text_for_lda

class TestADGPTFunctions(unittest.TestCase):

    @patch('app.load_llama_model')  # Mock the model loading
    def test_find_files(self, mock_load_llama_model):
        # Mock the model loading to avoid access violation
        mock_load_llama_model.return_value = (None, None)

        logging.info("Testing find_files with mock model loading.")
    
        with patch('os.walk') as mock_walk:
            mock_walk.return_value = [
                ('/path', [], ['places.txt', 'random_file.txt']),
                ('/path2', [], ['places.txt'])
            ]
            result = find_files('/test_path', 'places')
            self.assertEqual(result, ['/path/places.txt', '/path2/places.txt'])


    def test_get_lemma(self):
        self.assertEqual(get_lemma('running'), 'run')
        self.assertEqual(get_lemma('better'), 'good')
        self.assertEqual(get_lemma('data'), 'data')  # If no lemma exists, the word remains the same

    def test_get_time_difference(self):
        test_date = '2023-09-01'
        result = get_time_difference(test_date, news_type='bbc')
        self.assertIsInstance(result, int)
        self.assertTrue(result >= 0)

        result_2 = get_time_difference('March 1, 2023', news_type='AA')
        self.assertIsInstance(result_2, int)
        self.assertTrue(result_2 >= 0)

    def test_scrape_place_text(self):
        text = "I live in New York and visited Los Angeles."
        expected_cities = ['New York', 'Los Angeles']
        
        with patch('app.Nominatim') as mock_nominatim:
            mock_geolocator = mock_nominatim.return_value
            mock_geolocator.geocode.side_effect = [MagicMock(), MagicMock()]
            
            result = scrape_place_text(text)
            self.assertEqual(result, expected_cities)

    def test_prepare_text_for_lda(self):
        text = "I am studying Natural Language Processing and learning about LDA models."
        result = prepare_text_for_lda(text)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

if __name__ == '__main__':
    print("Starting tests...")
    unittest.main()

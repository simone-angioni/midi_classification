import main_classifier as classifier

classifier.classify("standard", 50, 80, "lstm-standard_50_feature_size_80_artist", True)
classifier.classify("standard", 50, 50, "lstm-standard_frame_50_feature_size_50_artist", True)
classifier.classify("standard", 50, 30, "lstm-standard_frame_50_feature_size_30_artist", True)

classifier.classify("full-standard", 50, 80, "lstm-full-standard_50_feature_size_80_artist", True)
classifier.classify("full-standard", 50, 50, "lstm-full-standard_frame_50_feature_size_50_artist", True)
classifier.classify("full-standard", 50, 30, "lstm-full-standard_frame_50_feature_size_30_artist", True)

classifier.classify("walk-forward", 50, 80, "lstm-walk-forward_50_feature_size_80_artist", True)
classifier.classify("walk-forward", 50, 50, "lstm-walk-forward_frame_50_feature_size_50_artist", True)
classifier.classify("walk-forward", 50, 30, "lstm-walk-forward_frame_50_feature_size_30_artist", True)

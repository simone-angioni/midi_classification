import main_classifier as classifier

classifier.classify("walk-forward", 50, 80, "shuffle_walk-forward_time_frame_50_feature_size_80", True)
classifier.classify("walk-forward", 50, 50, "shuffle_walk-forward_time_frame_50_feature_size_50", True)
classifier.classify("walk-forward", 50, 30, "shuffle_walk-forward_time_frame_50_feature_size_30", True)
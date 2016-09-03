// Copyright © 2016 Singulariti, Inc.
// All rights reserved.
// Created by Hao Chen (hao.chen@misingularity.io) on 6/5/2016

package com.singulariti.androidrnn;

import android.content.res.AssetManager;

public class TensorflowRNNJni {
    private static TensorflowRNNJni instance;

    public static TensorflowRNNJni getInstance(
            final AssetManager assetManager) {
        if (instance == null) {
            instance = new TensorflowRNNJni(assetManager);
        }

        return instance;
    }

    private static final String MODEL_FILE =
            "file:///android_asset/graph.pb";
    private static final String CHINESE_LABEL_FILE =
            "file:///android_asset/chinese.txt";
    private static final String PINYIN_LABEL_FILE =
            "file:///android_asset/pinyin.txt";

    private TensorflowRNNJni(final AssetManager assetManager) {
        initFromJNI(assetManager, MODEL_FILE, CHINESE_LABEL_FILE, PINYIN_LABEL_FILE);
    }

    public native void initFromJNI(
        AssetManager assetManager,
        String model,
        String chinese_labels,
        String pinyin_labels);

    public native String decode(String text);

    static {
        System.loadLibrary("tensorflow_rnn");
    }
}
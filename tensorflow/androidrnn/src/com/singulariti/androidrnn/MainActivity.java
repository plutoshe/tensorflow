// Copyright © 2016 Singulariti, Inc.
// All rights reserved.
// Created by Hao Chen (hao.chen@misingularity.io) on 6/5/2016

package com.singulariti.androidrnn;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import org.w3c.dom.Text;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        final EditText mEditText = (EditText) findViewById(R.id.edit_message);
        final TextView mTextView = (TextView) findViewById(R.id.text_view);
        Button mButton = (Button) findViewById(R.id.send_button);
        mButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String abc = TensorflowRNNJni.getInstance(getAssets()).decode(
                        mEditText.getText().toString());
                Log.e("wo", abc);
                mTextView.setText(abc);
            }
        });
    }
}

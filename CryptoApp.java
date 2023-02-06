import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button registerButton = findViewById(R.id.register_button);
        registerButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                EditText usernameEditText = findViewById(R.id.username_edit_text);
                EditText passwordEditText = findViewById(R.id.password_edit_text);
                String username = usernameEditText.getText().toString();
                String password = passwordEditText.getText().toString();
                
                if (!username.isEmpty() && !password.isEmpty()) {
                    // TODO: Register?
                    // server verification?

                    // Inflate the success message layout
                    LayoutInflater inflater = (LayoutInflater) getSystemService(Context.LAYOUT_INFLATER_SERVICE);
                    View successView = inflater.inflate(R.layout.success_message, container, false);

                    // Add the success message view to the container
                    container.addView(successView);
                } else {
                    Toast.makeText(MainActivity.this, "Please enter both username and password!", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }
}

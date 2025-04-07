from django import forms
from django.contrib.auth.models import User
from users.models import UserProfile, VideoUpload

class UserRegistrationForm(forms.ModelForm):
    name = forms.CharField(max_length=100, required=True, widget=forms.TextInput(attrs={'placeholder': 'Full Name'}))
    mobile = forms.CharField(max_length=15, required=True, widget=forms.TextInput(attrs={'placeholder': 'Mobile Number'}))
    dob = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}), required=True)
    password = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Password'}), required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password']

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data['password'])  # Hash password
        if commit:
            user.save()
            UserProfile.objects.create(
                user=user,
                name=self.cleaned_data['name'],
                mobile=self.cleaned_data['mobile'],
                dob=self.cleaned_data['dob'],
            )
        return user

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = VideoUpload
        fields = ['uploaded_video']

    def clean_upload_time(self):
        upload_time = self.cleaned_data.get('upload_time')
        if not upload_time:
            from django.utils.timezone import now
            upload_time = now()
        return upload_time

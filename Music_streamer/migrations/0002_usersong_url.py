# Generated by Django 3.2.9 on 2021-11-26 01:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Music_streamer', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='usersong',
            name='url',
            field=models.CharField(default='charField', max_length=400),
        ),
    ]